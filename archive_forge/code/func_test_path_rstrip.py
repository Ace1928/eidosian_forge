from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_path_rstrip(self):
    scheme = 'https'
    server = '13.37.73.31'
    path = 'images/ubuntu-14.04.vmdk/'
    dc_path = 'datacenter-1'
    ds_name = 'datastore-1'
    params = {'dcPath': dc_path, 'dsName': ds_name}
    query = urlparse.urlencode(params)
    url = datastore.DatastoreURL(scheme, server, path, dc_path, ds_name)
    expected_url = '%s://%s/folder/%s?%s' % (scheme, server, path.rstrip('/'), query)
    self.assertEqual(expected_url, str(url))