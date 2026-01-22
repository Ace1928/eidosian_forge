import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _set_operator_cloud(self, **kwargs):
    operator_config = self.config.get_one(cloud=self._op_name, **kwargs)
    self.operator_cloud = connection.Connection(config=operator_config)
    _disable_keep_alive(self.operator_cloud)