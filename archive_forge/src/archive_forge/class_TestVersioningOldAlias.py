from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
class TestVersioningOldAlias(TestVersioning):
    _set_ver_cmd = ['setversioning']
    _get_ver_cmd = ['getversioning']