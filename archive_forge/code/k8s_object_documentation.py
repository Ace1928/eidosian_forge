from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
Implements evaluation of `self[key] = value`.

    Args:
      key: value of the key field
      value: value of the value field

    Raises:
      KeyError: if a message with the same key value already exists, but is
        hidden by the filter func, this is raised to prevent accidental
        overwrites
    