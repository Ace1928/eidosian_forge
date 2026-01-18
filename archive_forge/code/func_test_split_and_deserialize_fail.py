import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_split_and_deserialize_fail(self):
    self.assertRaises(exc.CommandError, utils.split_and_deserialize, 'foo:bar')