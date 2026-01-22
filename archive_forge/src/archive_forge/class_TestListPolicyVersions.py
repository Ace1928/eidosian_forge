from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestListPolicyVersions(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<ListPolicyVersionsResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ListPolicyVersionsResult>\n    <Versions>\n      <member>\n        <IsDefaultVersion>false</IsDefaultVersion>\n        <VersionId>v3</VersionId>\n        <CreateDate>2014-09-17T22:32:43Z</CreateDate>\n      </member>\n      <member>\n        <IsDefaultVersion>true</IsDefaultVersion>\n        <VersionId>v2</VersionId>\n        <CreateDate>2014-09-15T20:31:47Z</CreateDate>\n      </member>\n      <member>\n        <IsDefaultVersion>false</IsDefaultVersion>\n        <VersionId>v1</VersionId>\n        <CreateDate>2014-09-15T17:36:14Z</CreateDate>\n      </member>\n    </Versions>\n    <IsTruncated>false</IsTruncated>\n  </ListPolicyVersionsResult>\n  <ResponseMetadata>\n    <RequestId>a31d1a86-3eba-11e4-9d0d-6f969EXAMPLE</RequestId>\n  </ResponseMetadata>\n</ListPolicyVersionsResponse>\n        '

    def test_list_policy_versions(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.list_policy_versions('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', max_items=3)
        self.assert_request_parameters({'Action': 'ListPolicyVersions', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'MaxItems': 3}, ignore_params_values=['Version'])
        self.assertEqual(len(response['list_policy_versions_response']['list_policy_versions_result']['versions']), 3)