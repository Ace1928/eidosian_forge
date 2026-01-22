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
def _make_conditions_from_filters(filters, is_public=None):
    filters = filters.copy()
    image_conditions = []
    prop_conditions = []
    tag_conditions = []
    if is_public is not None:
        if is_public:
            image_conditions.append(models.Image.visibility == 'public')
        else:
            image_conditions.append(models.Image.visibility != 'public')
    if 'os_hidden' in filters:
        os_hidden = filters.pop('os_hidden')
        image_conditions.append(models.Image.os_hidden == os_hidden)
    if 'checksum' in filters:
        checksum = filters.pop('checksum')
        image_conditions.append(models.Image.checksum == checksum)
    if 'os_hash_value' in filters:
        os_hash_value = filters.pop('os_hash_value')
        image_conditions.append(models.Image.os_hash_value == os_hash_value)
    for k, v in filters.pop('properties', {}).items():
        prop_filters = _make_image_property_condition(key=k, value=v)
        prop_conditions.append(prop_filters)
    if 'changes-since' in filters:
        changes_since = timeutils.normalize_time(filters.pop('changes-since'))
        image_conditions.append(models.Image.updated_at > changes_since)
    if 'deleted' in filters:
        deleted_filter = filters.pop('deleted')
        image_conditions.append(models.Image.deleted == deleted_filter)
        if not deleted_filter:
            image_statuses = [s for s in STATUSES if s != 'killed']
            image_conditions.append(models.Image.status.in_(image_statuses))
    if 'tags' in filters:
        tags = filters.pop('tags')
        for tag in tags:
            alias = sa_orm.aliased(models.ImageTag)
            tag_filters = [alias.deleted == False]
            tag_filters.extend([alias.value == tag])
            tag_conditions.append((alias, tag_filters))
    filters = {k: v for k, v in filters.items() if v is not None}
    keys = list(filters.keys())
    for k in keys:
        key = k
        if k.endswith('_min') or k.endswith('_max'):
            key = key[0:-4]
            try:
                v = int(filters.pop(k))
            except ValueError:
                msg = _('Unable to filter on a range with a non-numeric value.')
                raise exception.InvalidFilterRangeValue(msg)
            if k.endswith('_min'):
                image_conditions.append(getattr(models.Image, key) >= v)
            if k.endswith('_max'):
                image_conditions.append(getattr(models.Image, key) <= v)
        elif k in ['created_at', 'updated_at']:
            attr_value = getattr(models.Image, key)
            operator, isotime = utils.split_filter_op(filters.pop(k))
            try:
                parsed_time = timeutils.parse_isotime(isotime)
                threshold = timeutils.normalize_time(parsed_time)
            except ValueError:
                msg = _('Bad "%s" query filter format. Use ISO 8601 DateTime notation.') % k
                raise exception.InvalidParameterValue(msg)
            comparison = utils.evaluate_filter_op(attr_value, operator, threshold)
            image_conditions.append(comparison)
        elif k in ['name', 'id', 'status', 'container_format', 'disk_format']:
            attr_value = getattr(models.Image, key)
            operator, list_value = utils.split_filter_op(filters.pop(k))
            if operator == 'in':
                threshold = utils.split_filter_value_for_quotes(list_value)
                comparison = attr_value.in_(threshold)
                image_conditions.append(comparison)
            elif operator == 'eq':
                image_conditions.append(attr_value == list_value)
            else:
                msg = _("Unable to filter by unknown operator '%s'.") % operator
                raise exception.InvalidFilterOperatorValue(msg)
    for k, value in filters.items():
        if hasattr(models.Image, k):
            image_conditions.append(getattr(models.Image, k) == value)
        else:
            prop_filters = _make_image_property_condition(key=k, value=value)
            prop_conditions.append(prop_filters)
    return (image_conditions, prop_conditions, tag_conditions)