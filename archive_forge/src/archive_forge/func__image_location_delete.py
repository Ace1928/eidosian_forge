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
def _image_location_delete(context, session, image_id, location_id, status, delete_time=None):
    if status not in ('deleted', 'pending_delete'):
        msg = _("The status of deleted image location can only be set to 'pending_delete' or 'deleted'")
        raise exception.Invalid(msg)
    try:
        location_ref = session.query(models.ImageLocation).filter_by(id=location_id).filter_by(image_id=image_id).one()
        delete_time = delete_time or timeutils.utcnow()
        location_ref.update({'deleted': True, 'status': status, 'updated_at': delete_time, 'deleted_at': delete_time})
        location_ref.save(session=session)
    except sa_orm.exc.NoResultFound:
        msg = _('No location found with ID %(loc)s from image %(img)s') % dict(loc=location_id, img=image_id)
        LOG.warning(msg)
        raise exception.NotFound(msg)