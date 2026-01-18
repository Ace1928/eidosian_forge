from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.exc as sa_exc
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
import glance.db.sqlalchemy.metadef_api.utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
Delete a resource type or raise if not found or is protected