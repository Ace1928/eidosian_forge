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
def _GetIntAttribute(self, config_attr, required):
    """Gets the given attribute in integer form.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      int, The integer value of the attribute, or None if it is not set.
    """
    attr_value = self._LoadAttribute(config_attr, required)
    if attr_value is None:
        return None
    try:
        return int(attr_value)
    except ValueError:
        raise InvalidValueError('The attribute [{attr}] must have an integer value: [{value}]'.format(attr=config_attr, value=attr_value))