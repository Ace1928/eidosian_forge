import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
class TestObjectUploads(BaseTestObject):

    def setUp(self):
        super(TestObjectUploads, self).setUp()
        self.content = self.getUniqueString().encode('latin-1')
        self.object_file = tempfile.NamedTemporaryFile(delete=False)
        self.object_file.write(self.content)
        self.object_file.close()
        self.md5, self.sha256 = utils._get_file_hashes(self.object_file.name)
        self.endpoint = self.cloud.object_store.get_endpoint()

    def test_create_object(self):
        self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': 1000}, slo={'min_segment_size': 500})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name)
        self.assert_calls()

    def test_create_object_index_rax(self):
        self.register_uris([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object='index.html'), status_code=201, validate=dict(headers={'access-control-allow-origin': '*', 'content-type': 'text/html'}))])
        headers = {'access-control-allow-origin': '*', 'content-type': 'text/html'}
        self.cloud.create_object(self.container, name='index.html', data='', **headers)
        self.assert_calls()

    def test_create_directory_marker_object(self):
        self.register_uris([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={'content-type': 'application/directory'}))])
        self.cloud.create_directory_marker_object(container=self.container, name=self.object)
        self.assert_calls()

    def test_create_dynamic_large_object(self):
        max_file_size = 2
        min_file_size = 1
        uris_to_mock = [dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404)]
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}/{index:0>6}'.format(endpoint=self.endpoint, container=self.container, object=self.object, index=index), status_code=201) for index, offset in enumerate(range(0, len(self.content), max_file_size))])
        uris_to_mock.append(dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={'x-object-manifest': '{container}/{object}'.format(container=self.container, object=self.object), 'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})))
        self.register_uris(uris_to_mock)
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, use_slo=False)
        self.assert_calls(stop_after=3)
        for key, value in self.calls[-1]['headers'].items():
            self.assertEqual(value, self.adapter.request_history[-1].headers[key], 'header mismatch in manifest call')

    def test_create_static_large_object(self):
        max_file_size = 25
        min_file_size = 1
        uris_to_mock = [dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404)]
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}/{index:0>6}'.format(endpoint=self.endpoint, container=self.container, object=self.object, index=index), status_code=201, headers=dict(Etag='etag{index}'.format(index=index))) for index, offset in enumerate(range(0, len(self.content), max_file_size))])
        uris_to_mock.append(dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})))
        self.register_uris(uris_to_mock)
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
        self.assert_calls(stop_after=3)
        for key, value in self.calls[-1]['headers'].items():
            self.assertEqual(value, self.adapter.request_history[-1].headers[key], 'header mismatch in manifest call')
        base_object = '/{container}/{object}'.format(container=self.container, object=self.object)
        self.assertEqual([{'path': '{base_object}/000000'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag0'}, {'path': '{base_object}/000001'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag1'}, {'path': '{base_object}/000002'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag2'}, {'path': '{base_object}/000003'.format(base_object=base_object), 'size_bytes': len(self.object) - 75, 'etag': 'etag3'}], self.adapter.request_history[-1].json())

    def test_slo_manifest_retry(self):
        """
        Uploading the SLO manifest file should be retried up to 3 times before
        giving up. This test should succeed on the 3rd and final attempt.
        """
        max_file_size = 25
        min_file_size = 1
        uris_to_mock = [dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404)]
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}/{index:0>6}'.format(endpoint=self.endpoint, container=self.container, object=self.object, index=index), status_code=201, headers=dict(Etag='etag{index}'.format(index=index))) for index, offset in enumerate(range(0, len(self.content), max_file_size))])
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
        self.register_uris(uris_to_mock)
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
        self.assert_calls(stop_after=3)
        for key, value in self.calls[-1]['headers'].items():
            self.assertEqual(value, self.adapter.request_history[-1].headers[key], 'header mismatch in manifest call')
        base_object = '/{container}/{object}'.format(container=self.container, object=self.object)
        self.assertEqual([{'path': '{base_object}/000000'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag0'}, {'path': '{base_object}/000001'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag1'}, {'path': '{base_object}/000002'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag2'}, {'path': '{base_object}/000003'.format(base_object=base_object), 'size_bytes': len(self.object) - 75, 'etag': 'etag3'}], self.adapter.request_history[-1].json())

    def test_slo_manifest_fail(self):
        """
        Uploading the SLO manifest file should be retried up to 3 times before
        giving up. This test fails all 3 attempts and should verify that we
        delete uploaded segments that begin with the object prefix.
        """
        max_file_size = 25
        min_file_size = 1
        uris_to_mock = [dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404)]
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}/{index:0>6}'.format(endpoint=self.endpoint, container=self.container, object=self.object, index=index), status_code=201, headers=dict(Etag='etag{index}'.format(index=index))) for index, offset in enumerate(range(0, len(self.content), max_file_size))])
        uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
        uris_to_mock.extend([dict(method='GET', uri='{endpoint}/images?format=json&prefix={prefix}'.format(endpoint=self.endpoint, prefix=self.object), complete_qs=True, json=[{'content_type': 'application/octet-stream', 'bytes': 1437258240, 'hash': '249219347276c331b87bf1ac2152d9af', 'last_modified': '2015-02-16T17:50:05.289600', 'name': self.object}]), dict(method='HEAD', uri='{endpoint}/images/{object}'.format(endpoint=self.endpoint, object=self.object), headers={'X-Timestamp': '1429036140.50253', 'X-Trans-Id': 'txbbb825960a3243b49a36f-005a0dadaedfw1', 'Content-Length': '1290170880', 'Last-Modified': 'Tue, 14 Apr 2015 18:29:01 GMT', 'X-Object-Meta-x-sdk-autocreated': 'true', 'X-Object-Meta-X-Shade-Sha256': 'does not matter', 'X-Object-Meta-X-Shade-Md5': 'does not matter', 'Date': 'Thu, 16 Nov 2017 15:24:30 GMT', 'Accept-Ranges': 'bytes', 'X-Static-Large-Object': 'false', 'Content-Type': 'application/octet-stream', 'Etag': '249219347276c331b87bf1ac2152d9af'}), dict(method='DELETE', uri='{endpoint}/images/{object}'.format(endpoint=self.endpoint, object=self.object))])
        self.register_uris(uris_to_mock)
        self.cloud.image_api_use_tasks = True
        self.assertRaises(exceptions.SDKException, self.cloud.create_object, container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
        self.assert_calls(stop_after=3)

    def test_object_segment_retry_failure(self):
        max_file_size = 25
        min_file_size = 1
        self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404), dict(method='PUT', uri='{endpoint}/{container}/{object}/000000'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000001'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000002'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000003'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=501), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201)])
        self.assertRaises(exceptions.SDKException, self.cloud.create_object, container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
        self.assert_calls(stop_after=3)

    def test_object_segment_retries(self):
        max_file_size = 25
        min_file_size = 1
        self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404), dict(method='PUT', uri='{endpoint}/{container}/{object}/000000'.format(endpoint=self.endpoint, container=self.container, object=self.object), headers={'etag': 'etag0'}, status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000001'.format(endpoint=self.endpoint, container=self.container, object=self.object), headers={'etag': 'etag1'}, status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000002'.format(endpoint=self.endpoint, container=self.container, object=self.object), headers={'etag': 'etag2'}, status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000003'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=501), dict(method='PUT', uri='{endpoint}/{container}/{object}/000003'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, headers={'etag': 'etag3'}), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
        self.assert_calls(stop_after=3)
        for key, value in self.calls[-1]['headers'].items():
            self.assertEqual(value, self.adapter.request_history[-1].headers[key], 'header mismatch in manifest call')
        base_object = '/{container}/{object}'.format(container=self.container, object=self.object)
        self.assertEqual([{'path': '{base_object}/000000'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag0'}, {'path': '{base_object}/000001'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag1'}, {'path': '{base_object}/000002'.format(base_object=base_object), 'size_bytes': 25, 'etag': 'etag2'}, {'path': '{base_object}/000003'.format(base_object=base_object), 'size_bytes': len(self.object) - 75, 'etag': 'etag3'}], self.adapter.request_history[-1].json())

    def test_create_object_skip_checksum(self):
        self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': 1000}, slo={'min_segment_size': 500})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=200), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={}))])
        self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, generate_checksums=False)
        self.assert_calls()

    def test_create_object_data(self):
        self.register_uris([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={}, data=self.content))])
        self.cloud.create_object(container=self.container, name=self.object, data=self.content)
        self.assert_calls()