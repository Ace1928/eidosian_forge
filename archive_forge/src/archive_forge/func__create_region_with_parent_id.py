import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def _create_region_with_parent_id(self, parent_id=None):
    new_region = unit.new_region_ref(parent_region_id=parent_id)
    PROVIDERS.catalog_api.create_region(new_region)
    return new_region