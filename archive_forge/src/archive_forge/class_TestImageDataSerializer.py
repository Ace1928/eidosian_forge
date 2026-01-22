import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestImageDataSerializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageDataSerializer, self).setUp()
        self.serializer = glance.api.v2.image_data.ResponseSerializer()

    def test_download(self):
        request = wsgi.Request.blank('/')
        request.environ = {}
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        self.serializer.download(response, image)
        self.assertEqual(b'ZZZ', response.body)
        self.assertEqual('3', response.headers['Content-Length'])
        self.assertNotIn('Content-MD5', response.headers)
        self.assertEqual('application/octet-stream', response.headers['Content-Type'])

    def test_range_requests_for_image_downloads(self):
        """
        Test partial download 'Range' requests for images (random image access)
        """

        def download_successful_Range(d_range):
            request = wsgi.Request.blank('/')
            request.environ = {}
            request.headers['Range'] = d_range
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
            self.serializer.download(response, image)
            self.assertEqual(206, response.status_code)
            self.assertEqual('2', response.headers['Content-Length'])
            self.assertEqual('bytes 1-2/3', response.headers['Content-Range'])
            self.assertEqual(b'YZ', response.body)
        download_successful_Range('bytes=1-2')
        download_successful_Range('bytes=1-')
        download_successful_Range('bytes=1-3')
        download_successful_Range('bytes=-2')
        download_successful_Range('bytes=1-100')

        def full_image_download_w_range(d_range):
            request = wsgi.Request.blank('/')
            request.environ = {}
            request.headers['Range'] = d_range
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
            self.serializer.download(response, image)
            self.assertEqual(206, response.status_code)
            self.assertEqual('3', response.headers['Content-Length'])
            self.assertEqual('bytes 0-2/3', response.headers['Content-Range'])
            self.assertEqual(b'XYZ', response.body)
        full_image_download_w_range('bytes=0-')
        full_image_download_w_range('bytes=0-2')
        full_image_download_w_range('bytes=0-3')
        full_image_download_w_range('bytes=-3')
        full_image_download_w_range('bytes=-4')
        full_image_download_w_range('bytes=0-100')
        full_image_download_w_range('bytes=-100')

        def download_failures_Range(d_range):
            request = wsgi.Request.blank('/')
            request.environ = {}
            request.headers['Range'] = d_range
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
            self.assertRaises(webob.exc.HTTPRequestRangeNotSatisfiable, self.serializer.download, response, image)
            return
        download_failures_Range('bytes=4-1')
        download_failures_Range('bytes=4-')
        download_failures_Range('bytes=3-')
        download_failures_Range('bytes=1')
        download_failures_Range('bytes=100')
        download_failures_Range('bytes=100-')
        download_failures_Range('bytes=')

    def test_multi_range_requests_raises_bad_request_error(self):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Range'] = 'bytes=0-0,-1'
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        self.assertRaises(webob.exc.HTTPBadRequest, self.serializer.download, response, image)

    def test_download_failure_with_valid_range(self):
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
            mock_get_data.side_effect = glance_store.NotFound(image='image')
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Range'] = 'bytes=1-2'
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        image.get_data = mock_get_data
        self.assertRaises(webob.exc.HTTPNoContent, self.serializer.download, response, image)

    def test_content_range_requests_for_image_downloads(self):
        """
        Even though Content-Range is incorrect on requests, we support it
        for backward compatibility with clients written for pre-Pike
        Glance.
        The following test is for 'Content-Range' requests, which we have
        to ensure that we prevent regression.
        """

        def download_successful_ContentRange(d_range):
            request = wsgi.Request.blank('/')
            request.environ = {}
            request.headers['Content-Range'] = d_range
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
            self.serializer.download(response, image)
            self.assertEqual(206, response.status_code)
            self.assertEqual('2', response.headers['Content-Length'])
            self.assertEqual('bytes 1-2/3', response.headers['Content-Range'])
            self.assertEqual(b'YZ', response.body)
        download_successful_ContentRange('bytes 1-2/3')
        download_successful_ContentRange('bytes 1-2/*')

        def download_failures_ContentRange(d_range):
            request = wsgi.Request.blank('/')
            request.environ = {}
            request.headers['Content-Range'] = d_range
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
            self.assertRaises(webob.exc.HTTPRequestRangeNotSatisfiable, self.serializer.download, response, image)
            return
        download_failures_ContentRange('bytes -3/3')
        download_failures_ContentRange('bytes 1-/3')
        download_failures_ContentRange('bytes 1-3/3')
        download_failures_ContentRange('bytes 1-4/3')
        download_failures_ContentRange('bytes 1-4/*')
        download_failures_ContentRange('bytes 4-1/3')
        download_failures_ContentRange('bytes 4-1/*')
        download_failures_ContentRange('bytes 4-8/*')
        download_failures_ContentRange('bytes 4-8/10')
        download_failures_ContentRange('bytes 4-8/3')

    def test_download_failure_with_valid_content_range(self):
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
            mock_get_data.side_effect = glance_store.NotFound(image='image')
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Content-Range'] = 'bytes %s-%s/3' % (1, 2)
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        image.get_data = mock_get_data
        self.assertRaises(webob.exc.HTTPNoContent, self.serializer.download, response, image)

    def test_download_with_checksum(self):
        request = wsgi.Request.blank('/')
        request.environ = {}
        response = webob.Response()
        response.request = request
        checksum = '0745064918b49693cca64d6b6a13d28a'
        image = FakeImage(size=3, checksum=checksum, data=[b'Z', b'Z', b'Z'])
        self.serializer.download(response, image)
        self.assertEqual(b'ZZZ', response.body)
        self.assertEqual('3', response.headers['Content-Length'])
        self.assertEqual(checksum, response.headers['Content-MD5'])
        self.assertEqual('application/octet-stream', response.headers['Content-Type'])

    def test_download_forbidden(self):
        """Make sure the serializer can return 403 forbidden error instead of
        500 internal server error.
        """

        def get_data(*args, **kwargs):
            raise exception.Forbidden()
        self.mock_object(glance.domain.proxy.Image, 'get_data', get_data)
        request = wsgi.Request.blank('/')
        request.environ = {}
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=iter('ZZZ'))
        image.get_data = get_data
        self.assertRaises(webob.exc.HTTPForbidden, self.serializer.download, response, image)

    def test_download_no_content(self):
        """Test image download returns HTTPNoContent

        Make sure that serializer returns 204 no content error in case of
        image data is not available at specified location.
        """
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
            mock_get_data.side_effect = glance_store.NotFound(image='image')
            request = wsgi.Request.blank('/')
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=iter('ZZZ'))
            image.get_data = mock_get_data
            self.assertRaises(webob.exc.HTTPNoContent, self.serializer.download, response, image)

    def test_download_service_unavailable(self):
        """Test image download returns HTTPServiceUnavailable."""
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
            mock_get_data.side_effect = glance_store.RemoteServiceUnavailable()
            request = wsgi.Request.blank('/')
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=iter('ZZZ'))
            image.get_data = mock_get_data
            self.assertRaises(webob.exc.HTTPServiceUnavailable, self.serializer.download, response, image)

    def test_download_store_get_not_support(self):
        """Test image download returns HTTPBadRequest.

        Make sure that serializer returns 400 bad request error in case of
        getting images from this store is not supported at specified location.
        """
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
            mock_get_data.side_effect = glance_store.StoreGetNotSupported()
            request = wsgi.Request.blank('/')
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=iter('ZZZ'))
            image.get_data = mock_get_data
            self.assertRaises(webob.exc.HTTPBadRequest, self.serializer.download, response, image)

    def test_download_store_random_get_not_support(self):
        """Test image download returns HTTPBadRequest.

        Make sure that serializer returns 400 bad request error in case of
        getting randomly images from this store is not supported at
        specified location.
        """
        with mock.patch.object(glance.domain.proxy.Image, 'get_data') as m_get_data:
            err = glance_store.StoreRandomGetNotSupported(offset=0, chunk_size=0)
            m_get_data.side_effect = err
            request = wsgi.Request.blank('/')
            response = webob.Response()
            response.request = request
            image = FakeImage(size=3, data=iter('ZZZ'))
            image.get_data = m_get_data
            self.assertRaises(webob.exc.HTTPBadRequest, self.serializer.download, response, image)

    def test_upload(self):
        request = webob.Request.blank('/')
        request.environ = {}
        response = webob.Response()
        response.request = request
        self.serializer.upload(response, {})
        self.assertEqual(http.NO_CONTENT, response.status_int)
        self.assertEqual('0', response.headers['Content-Length'])

    def test_stage(self):
        request = webob.Request.blank('/')
        request.environ = {}
        response = webob.Response()
        response.request = request
        self.serializer.stage(response, {})
        self.assertEqual(http.NO_CONTENT, response.status_int)
        self.assertEqual('0', response.headers['Content-Length'])