from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
class ConfigParsingError(core_exceptions.Error):
    """An exception raised when parsing kubeconfig file."""