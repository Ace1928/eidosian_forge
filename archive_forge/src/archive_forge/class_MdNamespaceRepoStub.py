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
class MdNamespaceRepoStub(object):

    def add(self, namespace):
        return 'mdns_add'

    def get(self, namespace):
        return 'mdns_get'

    def list(self, *args, **kwargs):
        return ['mdns_list']

    def save(self, namespace):
        return 'mdns_save'

    def remove(self, namespace):
        return 'mdns_remove'

    def remove_tags(self, namespace):
        return 'mdtags_remove'