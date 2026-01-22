from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def _create_private_namespace(fn_call, data):
    path = '/v2/metadefs/namespaces'
    return fn_call(path=path, data=data)