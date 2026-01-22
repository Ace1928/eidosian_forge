import argparse
from unittest import mock
import uuid
import fixtures
from oslo_config import cfg
from keystoneclient.auth import base
from keystoneclient.auth import cli
from keystoneclient.tests.unit.auth import utils
class TesterPlugin(base.BaseAuthPlugin):

    def get_token(self, *args, **kwargs):
        return None

    @classmethod
    def get_options(cls):
        deprecated = [cfg.DeprecatedOpt('test-other')]
        return [cfg.StrOpt('test-opt', help='tester', deprecated_opts=deprecated)]