import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_to_dev(self):
    self.assertEqual(version.SemanticVersion(1, 2, 3, dev_count=1), version.SemanticVersion(1, 2, 3).to_dev(1))
    self.assertEqual(version.SemanticVersion(1, 2, 3, 'rc', 1, dev_count=1), version.SemanticVersion(1, 2, 3, 'rc', 1).to_dev(1))