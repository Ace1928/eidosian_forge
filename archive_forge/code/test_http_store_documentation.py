from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
Test http store raises RemoteServiceUnavailable.