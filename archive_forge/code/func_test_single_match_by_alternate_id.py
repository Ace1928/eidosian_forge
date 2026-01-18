import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_single_match_by_alternate_id(self):
    the_id = 'Richard'

    class Test(resource.Resource):
        other_id = resource.Body('other_id', alternate_id=True)
    match = Test(other_id=the_id)
    result = Test._get_one_match(the_id, [match])
    self.assertIs(result, match)