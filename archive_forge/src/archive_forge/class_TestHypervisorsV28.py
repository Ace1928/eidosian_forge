from novaclient.tests.functional.v2.legacy import test_hypervisors
class TestHypervisorsV28(test_hypervisors.TestHypervisors):
    COMPUTE_API_VERSION = '2.28'

    def test_list(self):
        self._test_list(dict)