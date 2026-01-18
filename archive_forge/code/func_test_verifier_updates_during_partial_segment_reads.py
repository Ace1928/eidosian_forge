import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import backend
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_verifier_updates_during_partial_segment_reads(self):
    self.reader.read(4)
    self.verifier.update.assert_called_once_with(b'1234')
    self.reader.seek(0)
    self.reader.read(2)
    self.verifier.update.assert_called_once_with(b'1234')
    self.reader.read(4)
    self.verifier.update.assert_called_with(b'56')
    self.assertEqual(2, self.verifier.update.call_count)