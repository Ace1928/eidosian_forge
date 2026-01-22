import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def _create_datastore(self, value):
    ds = vim_util.get_moref(value, 'Datastore')
    return ds