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
class MdTagRepoStub(object):

    def add(self, tag):
        return 'mdtag_add'

    def add_tags(self, tags, can_append=False):
        return ['mdtag_add_tags']

    def get(self, ns, tag_name):
        return 'mdtag_get'

    def list(self, *args, **kwargs):
        return ['mdtag_list']

    def save(self, tag):
        return 'mdtag_save'

    def remove(self, tag):
        return 'mdtag_remove'