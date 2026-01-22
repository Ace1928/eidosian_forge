from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.core import exceptions
class BuildpackBuilder(dataobject.DataObject):
    """Settings for building with a buildpack.

    Attributes:
      builder: Name of the builder.
      trust: True if the lifecycle should trust this builder.
      devmode: Build with devmode.
  """
    NAMES = ('builder', 'trust', 'devmode')