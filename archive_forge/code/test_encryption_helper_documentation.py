from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import six
import boto
from gslib.exception import CommandException
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.tests.util import SetBotoConfigForTest
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
Tests a config file with non-sequential decryption key numbering.