import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def _load_ks_cfg_opts(self):
    conf = cfg.ConfigOpts()
    for group, opts in self.oslo_config_dict.items():
        conf.register_group(cfg.OptGroup(group))
        if opts is not None:
            ks_loading.register_adapter_conf_options(conf, group)
            for name, val in opts.items():
                conf.set_override(name, val, group=group)
    return conf