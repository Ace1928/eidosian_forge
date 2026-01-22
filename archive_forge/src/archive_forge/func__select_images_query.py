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
def _select_images_query(context, session, image_conditions, admin_as_user, member_status, visibility):
    img_conditional_clause = sa_sql.and_(True, *image_conditions)
    regular_user = not context.is_admin or admin_as_user
    query_member = session.query(models.Image).join(models.Image.members).filter(img_conditional_clause)
    if regular_user:
        member_filters = [models.ImageMember.deleted == False]
        member_filters.extend([models.Image.visibility == 'shared'])
        if context.owner is not None:
            member_filters.extend([models.ImageMember.member == context.owner])
            if member_status != 'all':
                member_filters.extend([models.ImageMember.status == member_status])
        query_member = query_member.filter(sa_sql.and_(*member_filters))
    query_image = session.query(models.Image).filter(img_conditional_clause)
    if regular_user:
        visibility_filters = [models.Image.visibility == 'public', models.Image.visibility == 'community']
        query_image = query_image.filter(sa_sql.or_(*visibility_filters))
        query_image_owner = None
        if context.owner is not None:
            query_image_owner = session.query(models.Image).filter(models.Image.owner == context.owner).filter(img_conditional_clause)
        if query_image_owner is not None:
            query = query_image.union(query_image_owner, query_member)
        else:
            query = query_image.union(query_member)
        return query
    else:
        return query_image