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
def ValidateArgFromFile(arg_internal_name, arg_value):
    """Do checks/mutations on arg values parsed from YAML which need validation.

  Any arg not appearing in the _FILE_ARG_VALIDATORS dictionary is assumed to be
  a simple string to be validated by the default _ValidateString() function.

  Mutations of the args are done in limited cases to improve ease-of-use.
  This includes:
  1) The YAML parser automatically converts attribute values into numeric types
  where possible. The os-version-ids for Android devices happen to be integers,
  but the Testing service expects them to be strings, so we automatically
  convert them to strings so users don't have to quote each one.
  2) The include: keyword, plus all test args that normally expect lists (e.g.
  device-ids, os-version-ids, locales, orientations...), will also accept a
  single value which is not specified using YAML list notation (e.g not enclosed
  in []). Such single values are automatically converted into a list containing
  one element.

  Args:
    arg_internal_name: the internal form of the arg name.
    arg_value: the argument's value as parsed from the yaml file.

  Returns:
    The validated argument value.

  Raises:
    InvalidArgException: If the arg value is missing or is not valid.
  """
    if arg_value is None:
        raise test_exceptions.InvalidArgException(arg_internal_name, 'no argument value found.')
    if arg_internal_name in _FILE_ARG_VALIDATORS:
        return _FILE_ARG_VALIDATORS[arg_internal_name](arg_internal_name, arg_value)
    return _ValidateString(arg_internal_name, arg_value)