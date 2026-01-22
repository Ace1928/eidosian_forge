from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
class _PlaceholderQueueRef:
    """A placeholder class to simulate queue_ref resource objects used in CT APIs.

    This class simulates the behaviour of the resource object returned by
    tasks.parsers.ParseQueue(...) function. We use this placeholder class
    instead of creating an actual resource instance because otherwise it takes
    roughly 2 minutes to create resource instances for a 1000 queues.

    Attributes:
      _relative_path: A string representing the full path for a queue in the
        format: 'projects/<project>/locations/<location>/queues/<queue>'
    """

    def __init__(self, relative_path):
        """Initializes the instance and sets the relative path."""
        self._relative_path = relative_path

    def RelativeName(self):
        """Gets the string representing the full path for a queue.

      This is the only function we are currently using in CT APIs for the
      queue_ref resource object.

      Returns:
        A string representing the full path for a queue in the following
        format: 'projects/<project>/locations/<location>/queues/<queue>'
      """
        return self._relative_path