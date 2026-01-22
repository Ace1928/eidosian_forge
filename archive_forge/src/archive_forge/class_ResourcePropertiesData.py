import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class ResourcePropertiesData(BASE, HeatBase):
    """Represents resource properties data, current or older"""
    __tablename__ = 'resource_properties_data'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    data = sqlalchemy.Column('data', types.Json)
    encrypted = sqlalchemy.Column('encrypted', sqlalchemy.Boolean)