import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
Assert that correct size is returned even though 0 was provided.