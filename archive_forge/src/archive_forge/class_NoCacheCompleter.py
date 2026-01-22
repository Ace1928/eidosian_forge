from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.api_lib.util import resource_search
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import completion_cache
from googlecloudsdk.core.cache import resource_cache
import six
class NoCacheCompleter(six.with_metaclass(abc.ABCMeta, Converter)):
    """A completer that does not cache completions."""

    def __init__(self, cache=None, **kwargs):
        del cache
        super(NoCacheCompleter, self).__init__(**kwargs)

    @abc.abstractmethod
    def Complete(self, prefix, parameter_info):
        """Returns the list of strings matching prefix.

    This method is normally provided by the cache, but must be specified here
    in order to bypass the cache.

    Args:
      prefix: The resource prefix string to match.
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.

    Returns:
      The list of strings matching prefix.
    """
        del prefix
        del parameter_info

    def Update(self, parameter_info=None, aggregations=None):
        """Satisfies abc resolution and will never be called."""
        del parameter_info, aggregations