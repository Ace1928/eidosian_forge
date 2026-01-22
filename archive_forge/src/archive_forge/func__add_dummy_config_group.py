import copy
import ssl
import time
from unittest import mock
from dogpile.cache import proxy
from oslo_config import cfg
from oslo_utils import uuidutils
from pymemcache import KeepaliveOpts
from oslo_cache import _opts
from oslo_cache import core as cache
from oslo_cache import exception
from oslo_cache.tests import test_cache
def _add_dummy_config_group(self):
    self.config_fixture.register_opt(cfg.IntOpt('cache_time'), group=TEST_GROUP)
    self.config_fixture.register_opt(cfg.IntOpt('cache_time'), group=TEST_GROUP2)