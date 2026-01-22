import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class ModuleFinderTest(test_util.TestCase):

    def testFindMessage(self):
        self.assertEquals(descriptor.describe_message(descriptor.FileSet), descriptor.import_descriptor_loader('apitools.base.protorpclite.descriptor.FileSet'))

    def testFindField(self):
        self.assertEquals(descriptor.describe_field(descriptor.FileSet.files), descriptor.import_descriptor_loader('apitools.base.protorpclite.descriptor.FileSet.files'))

    def testFindEnumValue(self):
        self.assertEquals(descriptor.describe_enum_value(test_util.OptionalMessage.SimpleEnum.VAL1), descriptor.import_descriptor_loader('apitools.base.protorpclite.test_util.OptionalMessage.SimpleEnum.VAL1'))