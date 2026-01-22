from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
class MdResourceTypeRepoStub(object):

    def add(self, rt):
        return 'mdrt_add'

    def get(self, *args, **kwargs):
        return 'mdrt_get'

    def list(self, *args, **kwargs):
        return ['mdrt_list']

    def remove(self, *args, **kwargs):
        return 'mdrt_remove'