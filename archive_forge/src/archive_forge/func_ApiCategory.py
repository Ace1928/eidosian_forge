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
def ApiCategory(cls, api_category=None):
    """Returns the passed str if given, else the class API_CATEGORY."""
    return api_category if api_category is not None else cls.API_CATEGORY