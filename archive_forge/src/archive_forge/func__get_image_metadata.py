from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
def _get_image_metadata(self):
    return {'id': 'e31cb99c-fe89-49fb-9cc5-f5104fffa636'}