from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class NeutronBaseV2(_NeutronBase):

    @declarative.declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + 's'