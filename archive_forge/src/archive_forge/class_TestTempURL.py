from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
class TestTempURL(TestObjectStoreProxy):
    expires_iso8601_format = '%Y-%m-%dT%H:%M:%SZ'
    short_expires_iso8601_format = '%Y-%m-%d'
    time_errmsg = 'time must either be a whole number or in specific ISO 8601 format.'
    path_errmsg = 'path must be full path to an object e.g. /v1/a/c/o'
    url = '/v1/AUTH_account/c/o'
    seconds = 3600
    key = 'correcthorsebatterystaple'
    method = 'GET'
    expected_url = url + '?temp_url_sig=temp_url_signature&temp_url_expires=1400003600'
    expected_body = '\n'.join([method, '1400003600', url]).encode('utf-8')

    @mock.patch('hmac.HMAC')
    @mock.patch('time.time', return_value=1400000000)
    def test_generate_temp_url(self, time_mock, hmac_mock):
        hmac_mock().hexdigest.return_value = 'temp_url_signature'
        url = self.proxy.generate_temp_url(self.url, self.seconds, self.method, temp_url_key=self.key)
        key = self.key
        if not isinstance(key, bytes):
            key = key.encode('utf-8')
        self.assertEqual(url, self.expected_url)
        self.assertEqual(hmac_mock.mock_calls, [mock.call(), mock.call(key, self.expected_body, sha1), mock.call().hexdigest()])
        self.assertIsInstance(url, type(self.url))

    @mock.patch('hmac.HMAC')
    @mock.patch('time.time', return_value=1400000000)
    def test_generate_temp_url_ip_range(self, time_mock, hmac_mock):
        hmac_mock().hexdigest.return_value = 'temp_url_signature'
        ip_ranges = ['1.2.3.4', '1.2.3.4/24', '2001:db8::', b'1.2.3.4', b'1.2.3.4/24', b'2001:db8::']
        path = '/v1/AUTH_account/c/o/'
        expected_url = path + '?temp_url_sig=temp_url_signature&temp_url_expires=1400003600&temp_url_ip_range='
        for ip_range in ip_ranges:
            hmac_mock.reset_mock()
            url = self.proxy.generate_temp_url(path, self.seconds, self.method, temp_url_key=self.key, ip_range=ip_range)
            key = self.key
            if not isinstance(key, bytes):
                key = key.encode('utf-8')
            if isinstance(ip_range, bytes):
                ip_range_expected_url = expected_url + ip_range.decode('utf-8')
                expected_body = '\n'.join(['ip=' + ip_range.decode('utf-8'), self.method, '1400003600', path]).encode('utf-8')
            else:
                ip_range_expected_url = expected_url + ip_range
                expected_body = '\n'.join(['ip=' + ip_range, self.method, '1400003600', path]).encode('utf-8')
            self.assertEqual(url, ip_range_expected_url)
            self.assertEqual(hmac_mock.mock_calls, [mock.call(key, expected_body, sha1), mock.call().hexdigest()])
            self.assertIsInstance(url, type(path))

    @mock.patch('hmac.HMAC')
    def test_generate_temp_url_iso8601_argument(self, hmac_mock):
        hmac_mock().hexdigest.return_value = 'temp_url_signature'
        url = self.proxy.generate_temp_url(self.url, '2014-05-13T17:53:20Z', self.method, temp_url_key=self.key)
        self.assertEqual(url, self.expected_url)
        url = self.proxy.generate_temp_url(self.url, '2014-05-13T17:53:20Z', self.method, temp_url_key=self.key, absolute=True)
        self.assertEqual(url, self.expected_url)
        lt = time.localtime()
        expires = time.strftime(self.expires_iso8601_format[:-1], lt)
        if not isinstance(self.expected_url, str):
            expected_url = self.expected_url.replace(b'1400003600', bytes(str(int(time.mktime(lt))), encoding='ascii'))
        else:
            expected_url = self.expected_url.replace('1400003600', str(int(time.mktime(lt))))
        url = self.proxy.generate_temp_url(self.url, expires, self.method, temp_url_key=self.key)
        self.assertEqual(url, expected_url)
        expires = time.strftime(self.short_expires_iso8601_format, lt)
        lt = time.strptime(expires, self.short_expires_iso8601_format)
        if not isinstance(self.expected_url, str):
            expected_url = self.expected_url.replace(b'1400003600', bytes(str(int(time.mktime(lt))), encoding='ascii'))
        else:
            expected_url = self.expected_url.replace('1400003600', str(int(time.mktime(lt))))
        url = self.proxy.generate_temp_url(self.url, expires, self.method, temp_url_key=self.key)
        self.assertEqual(url, expected_url)

    @mock.patch('hmac.HMAC')
    @mock.patch('time.time', return_value=1400000000)
    def test_generate_temp_url_iso8601_output(self, time_mock, hmac_mock):
        hmac_mock().hexdigest.return_value = 'temp_url_signature'
        url = self.proxy.generate_temp_url(self.url, self.seconds, self.method, temp_url_key=self.key, iso8601=True)
        key = self.key
        if not isinstance(key, bytes):
            key = key.encode('utf-8')
        expires = time.strftime(self.expires_iso8601_format, time.gmtime(1400003600))
        if not isinstance(self.url, str):
            self.assertTrue(url.endswith(bytes(expires, 'utf-8')))
        else:
            self.assertTrue(url.endswith(expires))
        self.assertEqual(hmac_mock.mock_calls, [mock.call(), mock.call(key, self.expected_body, sha1), mock.call().hexdigest()])
        self.assertIsInstance(url, type(self.url))

    @mock.patch('hmac.HMAC')
    @mock.patch('time.time', return_value=1400000000)
    def test_generate_temp_url_prefix(self, time_mock, hmac_mock):
        hmac_mock().hexdigest.return_value = 'temp_url_signature'
        prefixes = ['', 'o', 'p0/p1/']
        for p in prefixes:
            hmac_mock.reset_mock()
            path = '/v1/AUTH_account/c/' + p
            expected_url = path + ('?temp_url_sig=temp_url_signature&temp_url_expires=1400003600&temp_url_prefix=' + p)
            expected_body = '\n'.join([self.method, '1400003600', 'prefix:' + path]).encode('utf-8')
            url = self.proxy.generate_temp_url(path, self.seconds, self.method, prefix=True, temp_url_key=self.key)
            key = self.key
            if not isinstance(key, bytes):
                key = key.encode('utf-8')
            self.assertEqual(url, expected_url)
            self.assertEqual(hmac_mock.mock_calls, [mock.call(key, expected_body, sha1), mock.call().hexdigest()])
            self.assertIsInstance(url, type(path))

    def test_generate_temp_url_invalid_path(self):
        self.assertRaisesRegex(ValueError, 'path must be representable as UTF-8', self.proxy.generate_temp_url, b'/v1/a/c/\xff', self.seconds, self.method, temp_url_key=self.key)

    @mock.patch('hmac.HMAC.hexdigest', return_value='temp_url_signature')
    def test_generate_absolute_expiry_temp_url(self, hmac_mock):
        if isinstance(self.expected_url, bytes):
            expected_url = self.expected_url.replace(b'1400003600', b'2146636800')
        else:
            expected_url = self.expected_url.replace(u'1400003600', u'2146636800')
        url = self.proxy.generate_temp_url(self.url, 2146636800, self.method, absolute=True, temp_url_key=self.key)
        self.assertEqual(url, expected_url)

    def test_generate_temp_url_bad_time(self):
        for bad_time in ['not_an_int', -1, 1.1, '-1', '1.1', '2015-05', '2015-05-01T01:00']:
            self.assertRaisesRegex(ValueError, self.time_errmsg, self.proxy.generate_temp_url, self.url, bad_time, self.method, temp_url_key=self.key)

    def test_generate_temp_url_bad_path(self):
        for bad_path in ['/v1/a/c', 'v1/a/c/o', 'blah/v1/a/c/o', '/v1//c/o', '/v1/a/c/', '/v1/a/c']:
            self.assertRaisesRegex(ValueError, self.path_errmsg, self.proxy.generate_temp_url, bad_path, 60, self.method, temp_url_key=self.key)