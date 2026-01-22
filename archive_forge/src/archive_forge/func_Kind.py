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
@classmethod
def Kind(cls, kind=None):
    """Returns the passed str if given, else the class KIND."""
    return kind if kind is not None else cls.KIND