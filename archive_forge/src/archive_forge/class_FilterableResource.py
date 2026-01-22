import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class FilterableResource(resource.Resource):
    allow_list = True
    base_path = '/fakes'
    _query_mapping = resource.QueryParameters('a')
    a = resource.Body('a')
    b = resource.Body('b')
    c = resource.Body('c')