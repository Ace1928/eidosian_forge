from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _FormatInvalidDimension(environment):
    """Return a human-readable string representing an invalid matrix dimension."""
    if getattr(environment, 'androidDevice', None) is not None:
        device = environment.androidDevice
        return '[OS-version {vers} on {model}]'.format(model=device.androidModelId, vers=device.androidVersionId)
    if getattr(environment, 'iosDevice', None) is not None:
        device = environment.iosDevice
        return '[OS-version {vers} on {model}]'.format(model=device.iosModelId, vers=device.iosVersionId)
    return '[unknown-environment]'