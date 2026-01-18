import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_resource_nested_identifier(self):
    rsrc_info = {'resource_name': 'aresource', 'links': [{'href': 'http://foo/name/id/resources/0', 'rel': 'self'}, {'href': 'http://foo/name/id', 'rel': 'stack'}, {'href': 'http://foo/n_name/n_id', 'rel': 'nested'}]}
    rsrc = hc_res.Resource(manager=None, info=rsrc_info)
    self.assertEqual('n_name/n_id', utils.resource_nested_identifier(rsrc))