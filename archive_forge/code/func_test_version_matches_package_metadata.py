from importlib import metadata
from absl.testing import absltest
import ml_dtypes
def test_version_matches_package_metadata(self):
    try:
        ml_dtypes_metadata = metadata.metadata('ml_dtypes')
    except ImportError as err:
        raise absltest.SkipTest('Package metadata not found') from err
    metadata_version = ml_dtypes_metadata['version']
    package_version = ml_dtypes.__version__
    self.assertEqual(metadata_version, package_version)