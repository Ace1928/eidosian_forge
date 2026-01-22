from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
def _BooleanValidator(attribute_name, attribute_value):
    """Validates boolean attributes.

  Args:
    attribute_name: str, the name of the attribute
    attribute_value: str | bool, the value of the attribute to validate

  Raises:
    InvalidValueError: if value is not boolean
  """
    accepted_strings = ['true', '1', 'on', 'yes', 'y', 'false', '0', 'off', 'no', 'n', '', 'none']
    if Stringize(attribute_value).lower() not in accepted_strings:
        raise InvalidValueError('The [{0}] value [{1}] is not valid. Possible values: [{2}]. (See http://yaml.org/type/bool.html)'.format(attribute_name, attribute_value, ', '.join([x if x else "''" for x in accepted_strings])))