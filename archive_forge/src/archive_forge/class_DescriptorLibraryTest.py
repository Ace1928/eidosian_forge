import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescriptorLibraryTest(test_util.TestCase):

    def setUp(self):
        self.packageless = descriptor.MessageDescriptor()
        self.packageless.name = 'Packageless'
        self.library = descriptor.DescriptorLibrary(descriptors={'not.real.Packageless': self.packageless, 'Packageless': self.packageless})

    def testLookupPackage(self):
        self.assertEquals('csv', self.library.lookup_package('csv'))
        self.assertEquals('apitools.base.protorpclite', self.library.lookup_package('apitools.base.protorpclite'))

    def testLookupNonPackages(self):
        lib = 'apitools.base.protorpclite.descriptor.DescriptorLibrary'
        for name in ('', 'a', lib):
            self.assertRaisesWithRegexpMatch(messages.DefinitionNotFoundError, 'Could not find definition for %s' % name, self.library.lookup_package, name)

    def testNoPackage(self):
        self.assertRaisesWithRegexpMatch(messages.DefinitionNotFoundError, 'Could not find definition for not.real', self.library.lookup_package, 'not.real.Packageless')
        self.assertEquals(None, self.library.lookup_package('Packageless'))