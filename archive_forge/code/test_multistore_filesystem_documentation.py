import builtins
import errno
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_config import cfg
from oslo_utils.secretutils import md5
from oslo_utils import units
import glance_store as store
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
Tests adding image with file permissions.

        Test that we can add an image via the filesystem backend with a
        required image file permission.
        