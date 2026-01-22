from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
class S3KeyTest(unittest.TestCase):
    s3 = True

    def setUp(self):
        self.conn = S3Connection()
        random.seed()
        self.bucket_name = 'keytest-%d-%d' % (time.time(), random.randint(1, 99999999))
        self.bucket = self.conn.create_bucket(self.bucket_name)

    def tearDown(self):
        for key in self.bucket:
            key.delete()
        self.bucket.delete()

    def test_set_contents_from_file_dataloss(self):
        content = 'abcde'
        sfp = StringIO()
        sfp.write(content)
        k = self.bucket.new_key('k')
        try:
            k.set_contents_from_file(sfp)
            self.fail('forgot to rewind so should fail.')
        except AttributeError:
            pass
        k.set_contents_from_file(sfp, rewind=True)
        self.assertEqual(k.size, 5)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content)
        sfp = StringIO()
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp)
        self.assertEqual(k.size, 0)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, '')

    def test_set_contents_as_file(self):
        content = '01234567890123456789'
        sfp = StringIO(content)
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp)
        self.assertEqual(k.size, 20)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content)
        sfp.seek(5)
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp)
        self.assertEqual(k.size, 15)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content[5:])
        sfp.seek(5)
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp, size=5)
        self.assertEqual(k.size, 5)
        self.assertEqual(sfp.tell(), 10)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content[5:10])

    def test_set_contents_with_md5(self):
        content = '01234567890123456789'
        sfp = StringIO(content)
        k = self.bucket.new_key('k')
        good_md5 = k.compute_md5(sfp)
        k.set_contents_from_file(sfp, md5=good_md5)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content)
        sfp.seek(5)
        k = self.bucket.new_key('k')
        good_md5 = k.compute_md5(sfp, size=5)
        k.set_contents_from_file(sfp, size=5, md5=good_md5)
        self.assertEqual(sfp.tell(), 10)
        kn = self.bucket.new_key('k')
        ks = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(ks, content[5:10])
        k = self.bucket.new_key('k')
        sfp.seek(0)
        hexdig, base64 = k.compute_md5(sfp)
        bad_md5 = (hexdig, base64[3:])
        try:
            k.set_contents_from_file(sfp, md5=bad_md5)
            self.fail('should fail with bad md5')
        except S3ResponseError:
            pass

    def test_get_contents_with_md5(self):
        content = '01234567890123456789'
        sfp = StringIO(content)
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp)
        kn = self.bucket.new_key('k')
        s = kn.get_contents_as_string().decode('utf-8')
        self.assertEqual(kn.md5, k.md5)
        self.assertEqual(s, content)

    def test_file_callback(self):

        def callback(wrote, total):
            self.my_cb_cnt += 1
            self.assertNotEqual(wrote, self.my_cb_last, 'called twice with same value')
            self.my_cb_last = wrote
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        sfp = StringIO('')
        k.set_contents_from_file(sfp, cb=callback, num_cb=10)
        self.assertEqual(self.my_cb_cnt, 1)
        self.assertEqual(self.my_cb_last, 0)
        sfp.close()
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback)
        self.assertEqual(self.my_cb_cnt, 1)
        self.assertEqual(self.my_cb_last, 0)
        content = '01234567890123456789'
        sfp = StringIO(content)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.set_contents_from_file(sfp, cb=callback, num_cb=10)
        self.assertEqual(self.my_cb_cnt, 2)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback).decode('utf-8')
        self.assertEqual(self.my_cb_cnt, 2)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=-1)
        self.assertEqual(self.my_cb_cnt, 11)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=-1).decode('utf-8')
        self.assertEqual(self.my_cb_cnt, 11)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=1)
        self.assertTrue(self.my_cb_cnt <= 2)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=1).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 2)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=2)
        self.assertTrue(self.my_cb_cnt <= 2)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=2).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 2)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=3)
        self.assertTrue(self.my_cb_cnt <= 3)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=3).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 3)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=4)
        self.assertTrue(self.my_cb_cnt <= 4)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=4).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 4)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=6)
        self.assertTrue(self.my_cb_cnt <= 6)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=6).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 6)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=10)
        self.assertTrue(self.my_cb_cnt <= 10)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=10).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 10)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)
        sfp.seek(0)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        k = self.bucket.new_key('k')
        k.BufferSize = 2
        k.set_contents_from_file(sfp, cb=callback, num_cb=1000)
        self.assertTrue(self.my_cb_cnt <= 1000)
        self.assertEqual(self.my_cb_last, 20)
        self.my_cb_cnt = 0
        self.my_cb_last = None
        s = k.get_contents_as_string(cb=callback, num_cb=1000).decode('utf-8')
        self.assertTrue(self.my_cb_cnt <= 1000)
        self.assertEqual(self.my_cb_last, 20)
        self.assertEqual(s, content)

    def test_website_redirects(self):
        self.bucket.configure_website('index.html')
        key = self.bucket.new_key('redirect-key')
        self.assertTrue(key.set_redirect('http://www.amazon.com/'))
        self.assertEqual(key.get_redirect(), 'http://www.amazon.com/')
        self.assertTrue(key.set_redirect('http://aws.amazon.com/'))
        self.assertEqual(key.get_redirect(), 'http://aws.amazon.com/')

    def test_website_redirect_none_configured(self):
        key = self.bucket.new_key('redirect-key')
        key.set_contents_from_string('')
        self.assertEqual(key.get_redirect(), None)

    def test_website_redirect_with_bad_value(self):
        self.bucket.configure_website('index.html')
        key = self.bucket.new_key('redirect-key')
        with self.assertRaises(key.provider.storage_response_error):
            key.set_redirect('ftp://ftp.example.org')
        with self.assertRaises(key.provider.storage_response_error):
            key.set_redirect('')

    def test_setting_date(self):
        key = self.bucket.new_key('test_date')
        key.set_metadata('date', '20130524T155935Z')
        key.set_contents_from_string('Some text here.')
        check = self.bucket.get_key('test_date')
        self.assertEqual(check.get_metadata('date'), u'20130524T155935Z')
        self.assertTrue('x-amz-meta-date' in check._get_remote_metadata())

    def test_header_casing(self):
        key = self.bucket.new_key('test_header_case')
        key.set_metadata('Content-type', 'application/json')
        key.set_metadata('Content-md5', 'XmUKnus7svY1frWsVskxXg==')
        key.set_contents_from_string('{"abc": 123}')
        check = self.bucket.get_key('test_header_case')
        self.assertEqual(check.content_type, 'application/json')

    def test_header_encoding(self):
        key = self.bucket.new_key('test_header_encoding')
        key.set_metadata('Cache-control', u'public, max-age=500')
        key.set_metadata('Test-Plus', u'A plus (+)')
        key.set_metadata('Content-disposition', u'filename=Schöne Zeit.txt')
        key.set_metadata('Content-Encoding', 'gzip')
        key.set_metadata('Content-Language', 'de')
        key.set_metadata('Content-Type', 'application/pdf')
        self.assertEqual(key.content_type, 'application/pdf')
        key.set_metadata('X-Robots-Tag', 'all')
        key.set_metadata('Expires', u'Thu, 01 Dec 1994 16:00:00 GMT')
        key.set_contents_from_string('foo')
        check = self.bucket.get_key('test_header_encoding')
        remote_metadata = check._get_remote_metadata()
        self.assertIn(check.cache_control, ('public,%20max-age=500', 'public, max-age=500'))
        self.assertIn(remote_metadata['cache-control'], ('public,%20max-age=500', 'public, max-age=500'))
        self.assertEqual(check.get_metadata('test-plus'), 'A plus (+)')
        self.assertEqual(check.content_disposition, 'filename=Sch%C3%B6ne Zeit.txt')
        self.assertEqual(remote_metadata['content-disposition'], 'filename=Sch%C3%B6ne Zeit.txt')
        self.assertEqual(check.content_encoding, 'gzip')
        self.assertEqual(remote_metadata['content-encoding'], 'gzip')
        self.assertEqual(check.content_language, 'de')
        self.assertEqual(remote_metadata['content-language'], 'de')
        self.assertEqual(check.content_type, 'application/pdf')
        self.assertEqual(remote_metadata['content-type'], 'application/pdf')
        self.assertEqual(check.x_robots_tag, 'all')
        self.assertEqual(remote_metadata['x-robots-tag'], 'all')
        self.assertEqual(check.expires, 'Thu, 01 Dec 1994 16:00:00 GMT')
        self.assertEqual(remote_metadata['expires'], 'Thu, 01 Dec 1994 16:00:00 GMT')
        expected = u'filename=Schöne Zeit.txt'
        if six.PY2:
            expected = expected.encode('utf-8')
        self.assertEqual(urllib.parse.unquote(check.content_disposition), expected)

    def test_set_contents_with_sse_c(self):
        content = '01234567890123456789'
        header = {'x-amz-server-side-encryption-customer-algorithm': 'AES256', 'x-amz-server-side-encryption-customer-key': 'MAAxAHQAZQBzAHQASwBlAHkAVABvAFMAUwBFAEMAIQA=', 'x-amz-server-side-encryption-customer-key-MD5': 'fUgCZDDh6bfEMuP2bN38mg=='}
        k = self.bucket.new_key('testkey_for_sse_c')
        k.set_contents_from_string(content, headers=header)
        kn = self.bucket.new_key('testkey_for_sse_c')
        ks = kn.get_contents_as_string(headers=header)
        self.assertEqual(ks, content.encode('utf-8'))