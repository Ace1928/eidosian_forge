import http.client as http
import time
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import requests
def verify_image_hashes_and_status(test_obj, image_id, checksum=None, os_hash_value=None, status=None, os_hash_algo='sha512', size=None):
    """Makes image-detail request and checks response.

    :param test_obj: The test object; expected to have _url() and
                     _headers() defined on it
    :param image_id: Image id to use in the request
    :param checksum: Expected checksum (default: None)
    :param os_hash_value: Expected multihash value (default: None)
    :param status: Expected status (default: None)
    :param os_hash_algo: Expected value of os_hash_algo; only checked when
                         os_hash_value is not None (default: 'sha512')
    """
    path = test_obj._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=test_obj._headers())
    test_obj.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    test_obj.assertEqual(checksum, image['checksum'])
    if os_hash_value:
        test_obj.assertEqual(str(os_hash_algo), image['os_hash_algo'])
    test_obj.assertEqual(os_hash_value, image['os_hash_value'])
    test_obj.assertEqual(status, image['status'])
    test_obj.assertEqual(size, image['size'])