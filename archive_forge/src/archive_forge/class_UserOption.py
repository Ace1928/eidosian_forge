import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
class UserOption(sql.ModelBase):
    __tablename__ = 'user_option'
    user_id = sql.Column(sql.String(64), sql.ForeignKey('user.id', ondelete='CASCADE'), nullable=False, primary_key=True)
    option_id = sql.Column(sql.String(4), nullable=False, primary_key=True)
    option_value = sql.Column(sql.JsonBlob, nullable=True)

    def __init__(self, option_id, option_value):
        self.option_id = option_id
        self.option_value = option_value