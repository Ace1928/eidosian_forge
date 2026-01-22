import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
class TransferTest(unittest.TestCase):

    def assertRangeAndContentRangeCompatible(self, request, response):
        request_prefix = 'bytes='
        self.assertIn('range', request.headers)
        self.assertTrue(request.headers['range'].startswith(request_prefix))
        request_range = request.headers['range'][len(request_prefix):]
        response_prefix = 'bytes '
        self.assertIn('content-range', response.info)
        response_header = response.info['content-range']
        self.assertTrue(response_header.startswith(response_prefix))
        response_range = response_header[len(response_prefix):].partition('/')[0]
        msg = 'Request range ({0}) not a prefix of response_range ({1})'.format(request_range, response_range)
        self.assertTrue(response_range.startswith(request_range), msg=msg)

    def testComputeEndByte(self):
        total_size = 100
        chunksize = 10
        download = transfer.Download.FromStream(six.StringIO(), chunksize=chunksize, total_size=total_size)
        self.assertEqual(chunksize - 1, download._Download__ComputeEndByte(0, end=50))

    def testComputeEndByteReturnNone(self):
        download = transfer.Download.FromStream(six.StringIO())
        self.assertIsNone(download._Download__ComputeEndByte(0, use_chunks=False))

    def testComputeEndByteNoChunks(self):
        total_size = 100
        download = transfer.Download.FromStream(six.StringIO(), chunksize=10, total_size=total_size)
        for end in (None, 1000):
            self.assertEqual(total_size - 1, download._Download__ComputeEndByte(0, end=end, use_chunks=False), msg='Failed on end={0}'.format(end))

    def testComputeEndByteNoTotal(self):
        download = transfer.Download.FromStream(six.StringIO())
        default_chunksize = download.chunksize
        for chunksize in (100, default_chunksize):
            download.chunksize = chunksize
            for start in (0, 10):
                self.assertEqual(download.chunksize + start - 1, download._Download__ComputeEndByte(start), msg='Failed on start={0}, chunksize={1}'.format(start, chunksize))

    def testComputeEndByteSmallTotal(self):
        total_size = 100
        download = transfer.Download.FromStream(six.StringIO(), total_size=total_size)
        for start in (0, 10):
            self.assertEqual(total_size - 1, download._Download__ComputeEndByte(start), msg='Failed on start={0}'.format(start))

    def testDownloadThenStream(self):
        bytes_http = object()
        http = object()
        download_stream = six.StringIO()
        download = transfer.Download.FromStream(download_stream, total_size=26)
        download.bytes_http = bytes_http
        base_url = 'https://part.one/'
        with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
            make_request.return_value = http_wrapper.Response(info={'content-range': 'bytes 0-25/26', 'status': http_client.OK}, content=string.ascii_lowercase, request_url=base_url)
            request = http_wrapper.Request(url='https://part.one/')
            download.InitializeDownload(request, http=http)
            self.assertEqual(1, make_request.call_count)
            received_request = make_request.call_args[0][1]
            self.assertEqual(base_url, received_request.url)
            self.assertRangeAndContentRangeCompatible(received_request, make_request.return_value)
        with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
            make_request.return_value = http_wrapper.Response(info={'status': http_client.REQUESTED_RANGE_NOT_SATISFIABLE}, content='error', request_url=base_url)
            download.StreamInChunks()
            self.assertEqual(1, make_request.call_count)
            received_request = make_request.call_args[0][1]
            self.assertEqual('bytes=26-', received_request.headers['range'])

    def testGetRange(self):
        for start_byte, end_byte in [(0, 25), (5, 15), (0, 0), (25, 25)]:
            bytes_http = object()
            http = object()
            download_stream = six.StringIO()
            download = transfer.Download.FromStream(download_stream, total_size=26, auto_transfer=False)
            download.bytes_http = bytes_http
            base_url = 'https://part.one/'
            with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
                make_request.return_value = http_wrapper.Response(info={'content-range': 'bytes %d-%d/26' % (start_byte, end_byte), 'status': http_client.OK}, content=string.ascii_lowercase[start_byte:end_byte + 1], request_url=base_url)
                request = http_wrapper.Request(url='https://part.one/')
                download.InitializeDownload(request, http=http)
                download.GetRange(start_byte, end_byte)
                self.assertEqual(1, make_request.call_count)
                received_request = make_request.call_args[0][1]
                self.assertEqual(base_url, received_request.url)
                self.assertRangeAndContentRangeCompatible(received_request, make_request.return_value)

    def testNonChunkedDownload(self):
        bytes_http = object()
        http = object()
        download_stream = six.StringIO()
        download = transfer.Download.FromStream(download_stream, total_size=52)
        download.bytes_http = bytes_http
        base_url = 'https://part.one/'
        with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
            make_request.return_value = http_wrapper.Response(info={'content-range': 'bytes 0-51/52', 'status': http_client.OK}, content=string.ascii_lowercase * 2, request_url=base_url)
            request = http_wrapper.Request(url='https://part.one/')
            download.InitializeDownload(request, http=http)
            self.assertEqual(1, make_request.call_count)
            received_request = make_request.call_args[0][1]
            self.assertEqual(base_url, received_request.url)
            self.assertRangeAndContentRangeCompatible(received_request, make_request.return_value)
            download_stream.seek(0)
            self.assertEqual(string.ascii_lowercase * 2, download_stream.getvalue())

    def testChunkedDownload(self):
        bytes_http = object()
        http = object()
        download_stream = six.StringIO()
        download = transfer.Download.FromStream(download_stream, chunksize=26, total_size=52)
        download.bytes_http = bytes_http

        def _ReturnBytes(unused_http, http_request, *unused_args, **unused_kwds):
            url = http_request.url
            if url == 'https://part.one/':
                return http_wrapper.Response(info={'content-location': 'https://part.two/', 'content-range': 'bytes 0-25/52', 'status': http_client.PARTIAL_CONTENT}, content=string.ascii_lowercase, request_url='https://part.one/')
            elif url == 'https://part.two/':
                return http_wrapper.Response(info={'content-range': 'bytes 26-51/52', 'status': http_client.OK}, content=string.ascii_uppercase, request_url='https://part.two/')
            else:
                self.fail('Unknown URL requested: %s' % url)
        with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
            make_request.side_effect = _ReturnBytes
            request = http_wrapper.Request(url='https://part.one/')
            download.InitializeDownload(request, http=http)
            self.assertEqual(2, make_request.call_count)
            for call in make_request.call_args_list:
                self.assertRangeAndContentRangeCompatible(call[0][1], _ReturnBytes(*call[0]))
            download_stream.seek(0)
            self.assertEqual(string.ascii_lowercase + string.ascii_uppercase, download_stream.getvalue())

    def testFinalizesTransferUrlIfClientPresent(self):
        """Tests download's enforcement of client custom endpoints."""
        mock_client = mock.Mock()
        fake_json_data = json.dumps({'auto_transfer': False, 'progress': 0, 'total_size': 0, 'url': 'url'})
        transfer.Download.FromData(six.BytesIO(), fake_json_data, client=mock_client)
        mock_client.FinalizeTransferUrl.assert_called_once_with('url')

    def testMultipartEncoding(self):
        test_cases = ['line one\nFrom \nline two', u'name,main_ingredient\nRäksmörgås,Räkor\nBaguette,Bröd']
        for upload_contents in test_cases:
            multipart_body = '{"body_field_one": 7}'
            upload_bytes = upload_contents.encode('ascii', 'backslashreplace')
            upload_config = base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path=u'/resumable/upload', simple_multipart=True, simple_path=u'/upload')
            url_builder = base_api._UrlBuilder('http://www.uploads.com')
            upload = transfer.Upload.FromStream(six.BytesIO(upload_bytes), 'text/plain', total_size=len(upload_bytes))
            http_request = http_wrapper.Request('http://www.uploads.com', headers={'content-type': 'text/plain'}, body=multipart_body)
            upload.ConfigureRequest(upload_config, http_request, url_builder)
            self.assertEqual('multipart', url_builder.query_params['uploadType'])
            rewritten_upload_contents = b'\n'.join(http_request.body.split(b'--')[2].splitlines()[1:])
            self.assertTrue(rewritten_upload_contents.endswith(upload_bytes))
            upload = transfer.Upload.FromStream(six.BytesIO(upload_bytes), 'text/plain', total_size=len(upload_bytes))
            http_request = http_wrapper.Request('http://www.uploads.com', headers={'content-type': 'text/plain'})
            upload.ConfigureRequest(upload_config, http_request, url_builder)
            self.assertEqual(url_builder.query_params['uploadType'], 'media')
            rewritten_upload_contents = http_request.body
            self.assertTrue(rewritten_upload_contents.endswith(upload_bytes))