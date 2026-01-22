from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
def GetPlatform():
    """Returns the platform to run on.

  If set by the user, returns whatever value they specified without any
  validation. If not set by the user, default to managed

  """
    return properties.VALUES.run.platform.Get()