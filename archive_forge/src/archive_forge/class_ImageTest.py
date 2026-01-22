import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
class ImageTest(testtools.TestCase):

    def setUp(self):
        super(ImageTest, self).setUp()
        self.api = utils.FakeAPI(fixtures)
        self.mgr = images.ImageManager(self.api)

    def test_delete(self):
        image = self.mgr.get('1')
        image.delete()
        expect = [('HEAD', '/v1/images/1', {}, None), ('HEAD', '/v1/images/1', {}, None), ('DELETE', '/v1/images/1', {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_update(self):
        image = self.mgr.get('1')
        image.update(name='image-5')
        expect = [('HEAD', '/v1/images/1', {}, None), ('HEAD', '/v1/images/1', {}, None), ('PUT', '/v1/images/1', {'x-image-meta-name': 'image-5', 'x-glance-registry-purge-props': 'false'}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_data(self):
        image = self.mgr.get('1')
        data = ''.join([b for b in image.data()])
        expect = [('HEAD', '/v1/images/1', {}, None), ('HEAD', '/v1/images/1', {}, None), ('GET', '/v1/images/1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('XXX', data)
        data = ''.join([b for b in image.data(do_checksum=False)])
        expect += [('GET', '/v1/images/1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('XXX', data)

    def test_data_with_wrong_checksum(self):
        image = self.mgr.get('2')
        data = ''.join([b for b in image.data(do_checksum=False)])
        expect = [('HEAD', '/v1/images/2', {}, None), ('HEAD', '/v1/images/2', {}, None), ('GET', '/v1/images/2', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('YYY', data)
        data = image.data()
        expect += [('GET', '/v1/images/2', {}, None)]
        self.assertEqual(expect, self.api.calls)
        try:
            data = ''.join([b for b in image.data()])
            self.fail('data did not raise an error.')
        except IOError as e:
            self.assertEqual(errno.EPIPE, e.errno)
            msg = 'was fd7c5c4fdaa97163ee4ba8842baa537a expected wrong'
            self.assertIn(msg, str(e))

    def test_data_with_checksum(self):
        image = self.mgr.get('3')
        data = ''.join([b for b in image.data(do_checksum=False)])
        expect = [('HEAD', '/v1/images/3', {}, None), ('HEAD', '/v1/images/3', {}, None), ('GET', '/v1/images/3', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('ZZZ', data)
        data = ''.join([b for b in image.data()])
        expect += [('GET', '/v1/images/3', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual('ZZZ', data)