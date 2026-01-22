from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from functools import wraps
import os.path
import random
import re
import shutil
import tempfile
import six
import boto
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
def MakeTempName(self, kind, prefix='', suffix=''):
    """Creates a temporary name that is most-likely unique.

    Args:
      kind (str): A string indicating what kind of test name this is.
      prefix (str): Prefix prepended to the temporary name.
      suffix (str): Suffix string appended to end of temporary name.

    Returns:
      (str) The temporary name. If `kind` was "bucket", the temporary name may
      have coerced this string, including the supplied `prefix`, such that it
      contains only characters that are valid across all supported storage
      providers (e.g. replacing "_" with "-", converting uppercase letters to
      lowercase, etc.).
    """
    name = '{prefix}gsutil-test-{method}-{kind}'.format(prefix=prefix, method=self.GetTestMethodName(), kind=kind)
    name = name[:MAX_BUCKET_LENGTH - 13]
    name = '{name}-{rand}'.format(name=name, rand=self.MakeRandomTestString())
    total_name_len = len(name) + len(suffix)
    if suffix:
        if kind == 'bucket' and total_name_len > MAX_BUCKET_LENGTH:
            self.fail('Tried to create a psuedo-random bucket name with a specific suffix, but the generated name was too long and there was not enough room for the suffix. Please use shorter strings or perform name randomization manually.\nRequested name: ' + name + suffix)
        name += suffix
    if kind == 'bucket':
        name = util.MakeBucketNameValid(name)
    return name