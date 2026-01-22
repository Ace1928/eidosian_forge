import fixtures
from openstackclient.tests.functional.volume import base
class BaseVolumeTests(base.BaseVolumeTests):
    """Base class for Volume functional tests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.haz_volume_v2 = cls.is_service_enabled('block-storage', '2.0')

    def setUp(self):
        super().setUp()
        if not self.haz_volume_v2:
            self.skipTest('No Volume v2 service present')
        ver_fixture = fixtures.EnvironmentVariable('OS_VOLUME_API_VERSION', '2')
        self.useFixture(ver_fixture)