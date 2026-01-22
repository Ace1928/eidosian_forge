from openstack.tests.functional import base
class BaseImageTest(base.BaseFunctionalTest):
    _wait_for_timeout_key = 'OPENSTACKSDK_FUNC_TEST_TIMEOUT_IMAGE'

    def setUp(self):
        super().setUp()
        self._set_user_cloud(image_api_version='2')
        self._set_operator_cloud(image_api_version='2')
        if not self.user_cloud.has_service('image', '2'):
            self.skipTest('image service not supported by cloud')