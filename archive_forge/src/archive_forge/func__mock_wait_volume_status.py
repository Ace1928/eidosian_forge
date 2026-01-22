import socket
from unittest import mock
import uuid
from cinderclient.v3 import client as cinderclient
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import strutils
from glance.common import wsgi
from glance.tests import functional
def _mock_wait_volume_status(self, volume, status_transition, status_expected):
    volume.status = status_expected
    return volume