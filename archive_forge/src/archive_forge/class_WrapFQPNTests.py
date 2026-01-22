import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class WrapFQPNTests(TestCase):
    """
    Tests that ensure L{wrapFQPN} loads the
    L{twisted.python.modules.PythonModule} or
    L{twisted.python.modules.PythonAttribute} for a given FQPN.
    """

    def setUp(self):
        from twisted.python.modules import PythonModule, PythonAttribute
        from .._discover import wrapFQPN, InvalidFQPN, NoModule, NoObject
        self.PythonModule = PythonModule
        self.PythonAttribute = PythonAttribute
        self.wrapFQPN = wrapFQPN
        self.InvalidFQPN = InvalidFQPN
        self.NoModule = NoModule
        self.NoObject = NoObject

    def assertModuleWrapperRefersTo(self, moduleWrapper, module):
        """
        Assert that a L{twisted.python.modules.PythonModule} refers to a
        particular Python module.
        """
        self.assertIsInstance(moduleWrapper, self.PythonModule)
        self.assertEqual(moduleWrapper.name, module.__name__)
        self.assertIs(moduleWrapper.load(), module)

    def assertAttributeWrapperRefersTo(self, attributeWrapper, fqpn, obj):
        """
        Assert that a L{twisted.python.modules.PythonAttribute} refers to a
        particular Python object.
        """
        self.assertIsInstance(attributeWrapper, self.PythonAttribute)
        self.assertEqual(attributeWrapper.name, fqpn)
        self.assertIs(attributeWrapper.load(), obj)

    def test_failsWithEmptyFQPN(self):
        """
        L{wrapFQPN} raises L{InvalidFQPN} when given an empty string.
        """
        with self.assertRaises(self.InvalidFQPN):
            self.wrapFQPN('')

    def test_failsWithBadDotting(self):
        """"
        L{wrapFQPN} raises L{InvalidFQPN} when given a badly-dotted
        FQPN.  (e.g., x..y).
        """
        for bad in ('.fails', 'fails.', 'this..fails'):
            with self.assertRaises(self.InvalidFQPN):
                self.wrapFQPN(bad)

    def test_singleModule(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the single module a dotless FQPN describes.
        """
        import os
        moduleWrapper = self.wrapFQPN('os')
        self.assertIsInstance(moduleWrapper, self.PythonModule)
        self.assertIs(moduleWrapper.load(), os)

    def test_failsWithMissingSingleModuleOrPackage(self):
        """
        L{wrapFQPN} raises L{NoModule} when given a dotless FQPN that does
        not refer to a module or package.
        """
        with self.assertRaises(self.NoModule):
            self.wrapFQPN('this is not an acceptable name!')

    def test_singlePackage(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the single package a dotless FQPN describes.
        """
        import xml
        self.assertModuleWrapperRefersTo(self.wrapFQPN('xml'), xml)

    def test_multiplePackages(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the deepest package described by dotted FQPN.
        """
        import xml.etree
        self.assertModuleWrapperRefersTo(self.wrapFQPN('xml.etree'), xml.etree)

    def test_multiplePackagesFinalModule(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the deepest module described by dotted FQPN.
        """
        import xml.etree.ElementTree
        self.assertModuleWrapperRefersTo(self.wrapFQPN('xml.etree.ElementTree'), xml.etree.ElementTree)

    def test_singleModuleObject(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonAttribute}
        referring to the deepest object an FQPN names, traversing one module.
        """
        import os
        self.assertAttributeWrapperRefersTo(self.wrapFQPN('os.path'), 'os.path', os.path)

    def test_multiplePackagesObject(self):
        """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonAttribute}
        referring to the deepest object described by an FQPN,
        descending through several packages.
        """
        import xml.etree.ElementTree
        import automat
        for fqpn, obj in [('xml.etree.ElementTree.fromstring', xml.etree.ElementTree.fromstring), ('automat.MethodicalMachine.__doc__', automat.MethodicalMachine.__doc__)]:
            self.assertAttributeWrapperRefersTo(self.wrapFQPN(fqpn), fqpn, obj)

    def test_failsWithMultiplePackagesMissingModuleOrPackage(self):
        """
        L{wrapFQPN} raises L{NoObject} when given an FQPN that contains a
        missing attribute, module, or package.
        """
        for bad in ('xml.etree.nope!', 'xml.etree.nope!.but.the.rest.is.believable'):
            with self.assertRaises(self.NoObject):
                self.wrapFQPN(bad)