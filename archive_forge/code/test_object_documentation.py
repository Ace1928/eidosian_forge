import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils

        Uploading the SLO manifest file should be retried up to 3 times before
        giving up. This test fails all 3 attempts and should verify that we
        delete uploaded segments that begin with the object prefix.
        