from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def MakeU2F(path):
    d = {}
    d['vendor_id'] = 4176
    d['product_id'] = 1031
    d['path'] = path
    d['usage'] = 1
    d['usage_page'] = 61904
    return d