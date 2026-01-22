from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def _ValidatePositiveInteger(arg_internal_name, arg_value):
    """Validates an argument which should be an integer > 0."""
    try:
        if isinstance(arg_value, int):
            return POSITIVE_INT_PARSER(str(arg_value))
    except arg_parsers.ArgumentTypeError as e:
        raise test_exceptions.InvalidArgException(arg_internal_name, six.text_type(e))
    raise test_exceptions.InvalidArgException(arg_internal_name, arg_value)