import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_print_image_virtual_size_not_available(self):
    image = {'id': '42', 'virtual_size': None}
    saved_stdout = sys.stdout
    try:
        sys.stdout = output_list = io.StringIO()
        utils.print_image(image)
    finally:
        sys.stdout = saved_stdout
    self.assertEqual('+--------------+---------------+\n| Property     | Value         |\n+--------------+---------------+\n| id           | 42            |\n| virtual_size | Not available |\n+--------------+---------------+\n', output_list.getvalue())