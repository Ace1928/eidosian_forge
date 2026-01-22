from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class HostMount(object):

    def __init__(self, key, mountInfo):
        self.key = key
        self.mountInfo = mountInfo