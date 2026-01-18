from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
Adding tests for container URI validation.

        Container URI validation is different from secret URI validation.
        That's why adding separate tests for code coverage.
        