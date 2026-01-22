import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def _get_server_console_output(self, server_id, length=None):
    output = self.compute.get_server_console_output(server=server_id, length=length)
    if 'output' in output:
        return output['output']