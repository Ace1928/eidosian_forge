import random
import string
import uuid
import warnings
import fixtures
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
import sqlalchemy
from sqlalchemy import exc as sqla_exc
from heat.common import context
from heat.db import api as db_api
from heat.db import models
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine import stack
from heat.engine import template
class AnyInstance(object):
    """Comparator for validating allowed instance type."""

    def __init__(self, allowed_type):
        self._allowed_type = allowed_type

    def __eq__(self, other):
        return isinstance(other, self._allowed_type)

    def __ne__(self, other):
        return not self.__eq__(other)