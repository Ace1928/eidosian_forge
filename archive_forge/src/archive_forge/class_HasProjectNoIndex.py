from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class HasProjectNoIndex(HasProject):
    """Project mixin, add to subclasses that have a user."""
    project_id = sa.Column(sa.String(db_const.PROJECT_ID_FIELD_SIZE))