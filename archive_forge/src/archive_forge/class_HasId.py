from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class HasId(object):
    """id mixin, add to subclasses that have an id."""
    id = sa.Column(sa.String(db_const.UUID_FIELD_SIZE), primary_key=True, default=uuidutils.generate_uuid)