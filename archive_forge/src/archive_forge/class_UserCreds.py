import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class UserCreds(BASE, HeatBase):
    """Represents user credentials.

    Also, mirrors the 'context' handed in by wsgi.
    """
    __tablename__ = 'user_creds'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    username = sqlalchemy.Column(sqlalchemy.String(255))
    password = sqlalchemy.Column(sqlalchemy.String(255))
    region_name = sqlalchemy.Column(sqlalchemy.String(255))
    decrypt_method = sqlalchemy.Column(sqlalchemy.String(64))
    tenant = sqlalchemy.Column(sqlalchemy.String(1024))
    auth_url = sqlalchemy.Column(sqlalchemy.Text)
    tenant_id = sqlalchemy.Column(sqlalchemy.String(256))
    trust_id = sqlalchemy.Column(sqlalchemy.String(255))
    trustor_user_id = sqlalchemy.Column(sqlalchemy.String(64))
    stack = relationship(Stack, backref=backref('user_creds'), cascade_backrefs=False)