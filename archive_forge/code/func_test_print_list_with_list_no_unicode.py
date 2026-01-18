import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_print_list_with_list_no_unicode(self):

    class Struct(object):

        def __init__(self, **entries):
            self.__dict__.update(entries)
    columns = ['ID', 'Tags']
    images = [Struct(**{'id': 'b8e1c57e-907a-4239-aed8-0df8e54b8d2d', 'tags': ['Name1', 'Tag_123', 'veeeery long']})]
    saved_stdout = sys.stdout
    try:
        sys.stdout = output_list = io.StringIO()
        utils.print_list(images, columns)
    finally:
        sys.stdout = saved_stdout
    self.assertEqual("+--------------------------------------+--------------------------------------+\n| ID                                   | Tags                                 |\n+--------------------------------------+--------------------------------------+\n| b8e1c57e-907a-4239-aed8-0df8e54b8d2d | ['Name1', 'Tag_123', 'veeeery long'] |\n+--------------------------------------+--------------------------------------+\n", output_list.getvalue())