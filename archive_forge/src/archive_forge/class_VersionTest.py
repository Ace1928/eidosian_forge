import pbr
from heat.tests import common
from heat import version
class VersionTest(common.HeatTestCase):

    def test_version(self):
        self.assertIsInstance(version.version_info, pbr.version.VersionInfo)