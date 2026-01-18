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