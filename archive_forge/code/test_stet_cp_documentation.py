from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import stat
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils import temporary_file_util
Tests that cp does not seek-ahead for bytes if file size will change.