from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
import six
class ReplicationError(Exception):
    """Represents an error that occurred while replicating a resource to a given location."""

    def __init__(self, location, message):
        self._message = 'Failed to replicate to location [{}]: {}'.format(location, message)
        super(ReplicationError, self).__init__(self._message)

    def __str__(self):
        return self._message