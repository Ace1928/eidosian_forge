from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from gslib.commands.rsync import _ComputeNeededFileChecksums
from gslib.commands.rsync import _NA
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.utils.hashing_helper import CalculateB64EncodedCrc32cFromContents
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
Tests that we compute all/only needed file checksums.