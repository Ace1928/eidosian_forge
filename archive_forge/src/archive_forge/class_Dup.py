import gc
from sqlalchemy.ext import declarative
from sqlalchemy import orm
import testtools
from neutron_lib.db import standard_attr
from neutron_lib.tests import _base as base
class Dup(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
    api_collections = ['my_resource']