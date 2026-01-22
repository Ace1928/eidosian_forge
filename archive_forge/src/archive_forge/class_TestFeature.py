import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
class TestFeature(tests.TestCase):

    def test_caching(self):
        """Feature._probe is called by the feature at most once."""

        class InstrumentedFeature(features.Feature):

            def __init__(self):
                super().__init__()
                self.calls = []

            def _probe(self):
                self.calls.append('_probe')
                return False
        feature = InstrumentedFeature()
        feature.available()
        self.assertEqual(['_probe'], feature.calls)
        feature.available()
        self.assertEqual(['_probe'], feature.calls)

    def test_named_str(self):
        """Feature.__str__ should thunk to feature_name()."""

        class NamedFeature(features.Feature):

            def feature_name(self):
                return 'symlinks'
        feature = NamedFeature()
        self.assertEqual('symlinks', str(feature))

    def test_default_str(self):
        """Feature.__str__ should default to __class__.__name__."""

        class NamedFeature(features.Feature):
            pass
        feature = NamedFeature()
        self.assertEqual('NamedFeature', str(feature))