import unittest
import six
from apitools.base.py.testing import mock
from samples.iam_sample.iam_v1 import iam_v1_client  # nopep8
from samples.iam_sample.iam_v1 import iam_v1_messages  # nopep8
class IamGenClientTest(unittest.TestCase):

    def setUp(self):
        self.mocked_iam_v1 = mock.Client(iam_v1_client.IamV1)
        self.mocked_iam_v1.Mock()
        self.addCleanup(self.mocked_iam_v1.Unmock)

    def testFlatPath(self):
        get_method_config = self.mocked_iam_v1.projects_serviceAccounts_keys.GetMethodConfig('Get')
        self.assertEquals('v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}', get_method_config.flat_path)
        self.assertEquals('v1/{+name}', get_method_config.relative_path)

    def testServiceAccountsKeysList(self):
        response_key = iam_v1_messages.ServiceAccountKey(name=u'test-key')
        self.mocked_iam_v1.projects_serviceAccounts_keys.List.Expect(iam_v1_messages.IamProjectsServiceAccountsKeysListRequest(name=u'test-service-account.'), iam_v1_messages.ListServiceAccountKeysResponse(keys=[response_key]))
        result = self.mocked_iam_v1.projects_serviceAccounts_keys.List(iam_v1_messages.IamProjectsServiceAccountsKeysListRequest(name=u'test-service-account.'))
        self.assertEquals([response_key], result.keys)