from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import exceptions
class InterfaceNotFoundError(exceptions.Error):
    """Raised when an interface is not found."""

    def __init__(self, name_list):
        error_msg = ('interface ' + ', '.join(['%s'] * len(name_list))) % tuple(name_list) + ' not found'
        super(InterfaceNotFoundError, self).__init__(error_msg)