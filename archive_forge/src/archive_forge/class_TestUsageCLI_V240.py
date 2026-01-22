from novaclient.tests.functional.v2.legacy import test_usage
class TestUsageCLI_V240(test_usage.TestUsageCLI):
    COMPUTE_API_VERSION = '2.40'