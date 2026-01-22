from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def api_class_call1(_self, *args, **kwargs):
    return (args, kwargs)