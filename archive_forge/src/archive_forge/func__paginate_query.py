import datetime
import itertools
import threading
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import session as oslo_db_session
from oslo_log import log as logging
from oslo_utils import excutils
import osprofiler.sqlalchemy
from retrying import retry
import sqlalchemy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy import MetaData, Table
import sqlalchemy.orm as sa_orm
from sqlalchemy import sql
import sqlalchemy.sql as sa_sql
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db.sqlalchemy.metadef_api import (resource_type
from glance.db.sqlalchemy.metadef_api import (resource_type_association
from glance.db.sqlalchemy.metadef_api import namespace as metadef_namespace_api
from glance.db.sqlalchemy.metadef_api import object as metadef_object_api
from glance.db.sqlalchemy.metadef_api import property as metadef_property_api
from glance.db.sqlalchemy.metadef_api import tag as metadef_tag_api
from glance.db.sqlalchemy import models
from glance.db import utils as db_utils
from glance.i18n import _, _LW, _LI, _LE
def _paginate_query(query, model, limit, sort_keys, marker=None, sort_dir=None, sort_dirs=None):
    """Returns a query with sorting / pagination criteria added.

    Pagination works by requiring a unique sort_key, specified by sort_keys.
    (If sort_keys is not unique, then we risk looping through values.)
    We use the last row in the previous page as the 'marker' for pagination.
    So we must return values that follow the passed marker in the order.
    With a single-valued sort_key, this would be easy: sort_key > X.
    With a compound-values sort_key, (k1, k2, k3) we must do this to repeat
    the lexicographical ordering:
    (k1 > X1) or (k1 == X1 && k2 > X2) or (k1 == X1 && k2 == X2 && k3 > X3)

    We also have to cope with different sort_directions.

    Typically, the id of the last row is used as the client-facing pagination
    marker, then the actual marker object must be fetched from the db and
    passed in to us as marker.

    :param query: the query object to which we should add paging/sorting
    :param model: the ORM model class
    :param limit: maximum number of items to return
    :param sort_keys: array of attributes by which results should be sorted
    :param marker: the last item of the previous page; we returns the next
                    results after this value.
    :param sort_dir: direction in which results should be sorted (asc, desc)
    :param sort_dirs: per-column array of sort_dirs, corresponding to sort_keys

    :rtype: sqlalchemy.orm.query.Query
    :returns: The query with sorting/pagination added.
    """
    if 'id' not in sort_keys:
        LOG.warning(_LW('Id not in sort_keys; is sort_keys unique?'))
    assert not (sort_dir and sort_dirs)
    if sort_dir is None:
        sort_dir = 'asc'
    if sort_dirs is None:
        sort_dirs = [sort_dir] * len(sort_keys)
    assert len(sort_dirs) == len(sort_keys)
    if len(sort_dirs) < len(sort_keys):
        sort_dirs += [sort_dir] * (len(sort_keys) - len(sort_dirs))
    for current_sort_key, current_sort_dir in zip(sort_keys, sort_dirs):
        sort_dir_func = {'asc': sqlalchemy.asc, 'desc': sqlalchemy.desc}[current_sort_dir]
        try:
            sort_key_attr = getattr(model, current_sort_key)
        except AttributeError:
            raise exception.InvalidSortKey()
        query = query.order_by(sort_dir_func(sort_key_attr))
    default = ''
    if marker is not None:
        marker_values = []
        for sort_key in sort_keys:
            v = getattr(marker, sort_key)
            if v is None:
                v = default
            marker_values.append(v)
        criteria_list = []
        for i in range(len(sort_keys)):
            crit_attrs = []
            for j in range(i):
                model_attr = getattr(model, sort_keys[j])
                default = _get_default_column_value(model_attr.property.columns[0].type)
                attr = sa_sql.expression.case((model_attr != None, model_attr), else_=default)
                crit_attrs.append(attr == marker_values[j])
            model_attr = getattr(model, sort_keys[i])
            default = _get_default_column_value(model_attr.property.columns[0].type)
            attr = sa_sql.expression.case((model_attr != None, model_attr), else_=default)
            if sort_dirs[i] == 'desc':
                crit_attrs.append(attr < marker_values[i])
            elif sort_dirs[i] == 'asc':
                crit_attrs.append(attr > marker_values[i])
            else:
                raise ValueError(_("Unknown sort direction, must be 'desc' or 'asc'"))
            criteria = sa_sql.and_(*crit_attrs)
            criteria_list.append(criteria)
        f = sa_sql.or_(*criteria_list)
        query = query.filter(f)
    if limit is not None:
        query = query.limit(limit)
    return query