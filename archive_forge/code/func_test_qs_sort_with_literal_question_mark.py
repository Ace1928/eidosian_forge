import hashlib
import io
from unittest import mock
import uuid
from oslo_utils import secretutils
from oslo_utils import units
from oslo_vmware import api
from oslo_vmware import exceptions as vmware_exceptions
from oslo_vmware.objects import datacenter as oslo_datacenter
from oslo_vmware.objects import datastore as oslo_datastore
import glance_store._drivers.vmware_datastore as vm_store
from glance_store import backend
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_qs_sort_with_literal_question_mark(self):
    url = 'scheme://example.com/path?key2=val2&key1=val1?sort=true'
    exp_url = 'scheme://example.com/path?key1=val1%3Fsort%3Dtrue&key2=val2'
    self.assertEqual(exp_url, utils.sort_url_by_qs_keys(url))