from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import os
import re
import textwrap
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding as api_encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import encoding
def _ClearFields(fields, path_deque, py_dict):
    """Clear the given fields in a dict at a given path.

  Args:
    fields: A list of fields to clear
    path_deque: A deque containing path segments
    py_dict: A nested dict from which to clear the fields
  """
    tmp_dict = py_dict
    for elem in path_deque:
        tmp_dict = tmp_dict[elem]
    for field in fields:
        if field in tmp_dict:
            del tmp_dict[field]