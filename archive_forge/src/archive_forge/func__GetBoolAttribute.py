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
def _GetBoolAttribute(self, config_attr, required, validate=True):
    """Gets the given attribute in bool form.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.
      validate: bool, True to validate the value

    Returns:
      bool, The value of the attribute, or None if it is not set.
    """
    attr_value = self._LoadAttribute(config_attr, required)
    if validate:
        _BooleanValidator(config_attr, attr_value)
    if attr_value is None:
        return None
    attr_string_value = Stringize(attr_value).lower()
    if attr_string_value == 'none':
        return None
    return attr_string_value in ['1', 'true', 'on', 'yes', 'y']