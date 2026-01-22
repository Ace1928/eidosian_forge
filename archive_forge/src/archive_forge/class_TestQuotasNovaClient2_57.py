from novaclient.tests.functional.v2.legacy import test_quotas
class TestQuotasNovaClient2_57(TestQuotasNovaClient2_35):
    """Nova quotas functional tests."""
    COMPUTE_API_VERSION = '2.latest'
    _quota_resources = ['instances', 'cores', 'ram', 'metadata_items', 'key_pairs', 'server_groups', 'server_group_members']