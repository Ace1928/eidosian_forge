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
class ResourceCompleter(Converter):
    """A parsed resource parameter initializer.

  Attributes:
    collection_info: The resource registry collection info.
    parse: The resource URI parse function. Converts a URI string into a list
      of parsed parameters.
  """

    def __init__(self, collection=None, api_version=None, param=None, **kwargs):
        """Constructor.

    Args:
      collection: The resource collection name.
      api_version: The API version for collection, None for the default version.
      param: The updated parameter column name.
      **kwargs: Base class kwargs.
    """
        self.api_version = api_version
        if collection:
            self.collection_info = resources.REGISTRY.GetCollectionInfo(collection, api_version=api_version)
            params = self.collection_info.GetParams('')
            log.info('cache collection=%s api_version=%s params=%s' % (collection, self.collection_info.api_version, params))
            parameters = [resource_cache.Parameter(name=name, column=column) for column, name in enumerate(params)]
            parse = resources.REGISTRY.Parse

            def _Parse(string):
                return parse(string, collection=collection, enforce_collection=False, validate=False).AsDict()
            self.parse = _Parse
        else:
            params = []
            parameters = []
        super(ResourceCompleter, self).__init__(collection=collection, columns=len(params), column=params.index(param) if param else 0, parameters=parameters, **kwargs)