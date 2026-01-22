from openstack.tests.functional import base
class BaseBlockStorageTest(base.BaseFunctionalTest):
    _wait_for_timeout_key = 'OPENSTACKSDK_FUNC_TEST_TIMEOUT_BLOCK_STORAGE'

    def setUp(self):
        super(BaseBlockStorageTest, self).setUp()
        self._set_user_cloud(block_storage_api_version='2')
        self._set_operator_cloud(block_storage_api_version='2')
        if not self.user_cloud.has_service('block-storage', '2'):
            self.skipTest('block-storage service not supported by cloud')