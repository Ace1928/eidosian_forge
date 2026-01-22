import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class RawTemplateFiles(BASE, HeatBase):
    """Where template files json dicts are stored."""
    __tablename__ = 'raw_template_files'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    files = sqlalchemy.Column(types.Json)