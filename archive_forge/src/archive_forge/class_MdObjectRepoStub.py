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
class MdObjectRepoStub(object):

    def add(self, obj):
        return 'mdobj_add'

    def get(self, ns, obj_name):
        return 'mdobj_get'

    def list(self, *args, **kwargs):
        return ['mdobj_list']

    def save(self, obj):
        return 'mdobj_save'

    def remove(self, obj):
        return 'mdobj_remove'