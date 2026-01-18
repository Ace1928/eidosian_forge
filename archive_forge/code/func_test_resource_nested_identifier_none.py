import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_resource_nested_identifier_none(self):
    rsrc_info = {'resource_name': 'aresource', 'links': [{'href': 'http://foo/name/id/resources/0', 'rel': 'self'}, {'href': 'http://foo/name/id', 'rel': 'stack'}]}
    rsrc = hc_res.Resource(manager=None, info=rsrc_info)
    self.assertIsNone(utils.resource_nested_identifier(rsrc))