import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
def _test_options_get_one_cloud(self, test_opts):
    """Test options sent "to openstack.config"""
    for opt in test_opts.keys():
        if not test_opts[opt][1]:
            continue
        key = opt2attr(opt)
        if isinstance(test_opts[opt][0], str):
            cmd = opt + ' ' + test_opts[opt][0]
        else:
            cmd = opt
        kwargs = {key: test_opts[opt][0]}
        self._assert_cloud_region_arg(cmd, kwargs)