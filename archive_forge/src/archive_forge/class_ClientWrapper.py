from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
from dulwich import client
from dulwich import errors
from dulwich import index
from dulwich import porcelain
from dulwich import repo
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
class ClientWrapper(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for a git client wrapper.

  This provides a uniform interface around the various types of git clients
  from dulwich 0.10.1.
  """

    def __init__(self, transport, path):
        self._transport = transport
        self._path = path

    @abc.abstractmethod
    def GetRefs(self):
        """Get a dictionary of all refs available from the repository.

    Returns:
      ({str: str, ...}) Dictionary mapping ref names to commit ids.
    """
        pass