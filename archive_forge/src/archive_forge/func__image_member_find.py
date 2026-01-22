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
def _image_member_find(context, session, image_id=None, member=None, status=None, include_deleted=False):
    query = session.query(models.ImageMember)
    if not include_deleted:
        query = query.filter_by(deleted=False)
    if not context.is_admin:
        query = query.join(models.Image)
        filters = [models.Image.owner == context.owner, models.ImageMember.member == context.owner]
        query = query.filter(sa_sql.or_(*filters))
    if image_id is not None:
        query = query.filter(models.ImageMember.image_id == image_id)
    if member is not None:
        query = query.filter(models.ImageMember.member == member)
    if status is not None:
        query = query.filter(models.ImageMember.status == status)
    return query.all()