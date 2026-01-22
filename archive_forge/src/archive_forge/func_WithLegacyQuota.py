from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
@contextlib.contextmanager
def WithLegacyQuota():
    """Context where x-goog-user-project header is not populated.

  In this context, user's setting of quota_project will be ignored, including
  settings in:
  - --billing-project
  - configuration of billing/quota_project
  - CLOUDSDK_BILLING_QUOTA_PROJECT
  quota_project settings by the command group's Filter functions will also be
  ignored. Use this context when you don't want to send the x-goog-user-project
  header.

  Yields:
    yield to the surrounded code block.
  """
    properties.VALUES.PushInvocationValues()
    properties.VALUES.SetInvocationValue(properties.VALUES.billing.quota_project, properties.VALUES.billing.LEGACY, None)
    try:
        yield
    finally:
        properties.VALUES.PopInvocationValues()