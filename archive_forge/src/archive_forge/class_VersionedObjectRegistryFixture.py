from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
class VersionedObjectRegistryFixture(fixtures.Fixture):
    """Use a VersionedObjectRegistry as a temp registry pattern fixture.

    The pattern solution is to backup the object registry, register
    a class locally, and then restore the original registry. This could be
    used for test objects that do not need to be registered permanently but
    will have calls which lookup registration.
    """

    def setUp(self):
        super(VersionedObjectRegistryFixture, self).setUp()
        self._base_test_obj_backup = copy.deepcopy(base.VersionedObjectRegistry._registry._obj_classes)
        self.addCleanup(self._restore_obj_registry)

    @staticmethod
    def register(cls_name):
        base.VersionedObjectRegistry.register(cls_name)

    def _restore_obj_registry(self):
        base.VersionedObjectRegistry._registry._obj_classes = self._base_test_obj_backup