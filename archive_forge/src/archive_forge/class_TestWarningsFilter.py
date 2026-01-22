import warnings
import testtools
import fixtures
class TestWarningsFilter(testtools.TestCase, fixtures.TestWithFixtures):

    def test_filter(self):
        fixture = fixtures.WarningsFilter([{'action': 'ignore', 'category': DeprecationWarning}, {'action': 'once', 'category': UserWarning}])
        self.useFixture(fixture)
        with warnings.catch_warnings(record=True) as w:
            warnings.warn('deprecated', DeprecationWarning)
            warnings.warn('user', UserWarning)
        self.assertEqual(1, len(w))

    def test_filters_restored(self):

        class CustomWarning(Warning):
            pass
        fixture = fixtures.WarningsFilter([{'action': 'once', 'category': CustomWarning}])
        old_filters = warnings.filters[:]
        with fixture:
            new_filters = warnings.filters[:]
            self.assertEqual(len(old_filters) + 1, len(new_filters))
            self.assertNotEqual(old_filters, new_filters)
        new_filters = warnings.filters[:]
        self.assertEqual(len(old_filters), len(new_filters))
        self.assertEqual(old_filters, new_filters)