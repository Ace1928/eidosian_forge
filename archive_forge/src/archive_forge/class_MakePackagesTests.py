import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
class MakePackagesTests(TestCase):
    """
    Tests for L{_makePackages}, a helper for populating C{sys.modules} with
    fictional modules.
    """

    def test_nonModule(self):
        """
        A non-C{dict} value in the attributes dictionary passed to L{_makePackages}
        is preserved unchanged in the return value.
        """
        modules = {}
        _makePackages(None, dict(reactor='reactor'), modules)
        self.assertEqual(modules, dict(reactor='reactor'))

    def test_moduleWithAttribute(self):
        """
        A C{dict} value in the attributes dictionary passed to L{_makePackages}
        is turned into a L{ModuleType} instance with attributes populated from
        the items of that C{dict} value.
        """
        modules = {}
        _makePackages(None, dict(twisted=dict(version='123')), modules)
        self.assertIsInstance(modules, dict)
        self.assertIsInstance(modules['twisted'], ModuleType)
        self.assertEqual('twisted', modules['twisted'].__name__)
        self.assertEqual('123', modules['twisted'].version)

    def test_packageWithModule(self):
        """
        Processing of the attributes dictionary is recursive, so a C{dict} value
        it contains may itself contain a C{dict} value to the same effect.
        """
        modules = {}
        _makePackages(None, dict(twisted=dict(web=dict(version='321'))), modules)
        self.assertIsInstance(modules, dict)
        self.assertIsInstance(modules['twisted'], ModuleType)
        self.assertEqual('twisted', modules['twisted'].__name__)
        self.assertIsInstance(modules['twisted'].web, ModuleType)
        self.assertEqual('twisted.web', modules['twisted'].web.__name__)
        self.assertEqual('321', modules['twisted'].web.version)