from aodhclient.tests.functional import base
class CapabilitiesClientTest(base.ClientTestBase):

    def test_capabilities_scenario(self):
        result = self.aodh('capabilities', params='list')
        caps = self.parser.listing(result)[0]
        self.assertIsNotNone(caps)
        self.assertEqual('alarm_storage', caps['Field'])