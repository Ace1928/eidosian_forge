import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _get_resource_value(resource_key):
    return TEST_CONFIG.get_extra_config('functional').get(resource_key)