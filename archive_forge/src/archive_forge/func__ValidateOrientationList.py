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
def _ValidateOrientationList(arg_internal_name, arg_value):
    """Validates that 'orientations' only contains allowable values."""
    arg_value = ValidateStringList(arg_internal_name, arg_value)
    for orientation in arg_value:
        _ValidateOrientation(orientation)
    if len(arg_value) != len(set(arg_value)):
        raise test_exceptions.InvalidArgException(arg_internal_name, 'orientations may not be repeated.')
    return arg_value