from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import walker
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.cache import resource_cache
import six
class GetCache(object):
    """Context manager for opening a cache given a cache identifier name."""
    _TYPES = {'file': file_cache.Cache, 'resource': resource_cache.ResourceCache}

    def __init__(self, name, create=False):
        """Constructor.

    Args:
      name: The cache name to operate on. May be prefixed by "resource://" for
        resource cache names or "file://" for persistent file cache names. If
        only the prefix is specified then the default cache name for that prefix
        is used.
      create: Creates the persistent cache if it exists if True.

    Raises:
      CacheNotFound: If the cache does not exist.

    Returns:
      The cache object.
    """
        self._name = name
        self._create = create
        self._cache = None

    def _OpenCache(self, cache_class, name):
        try:
            return cache_class(name, create=self._create)
        except cache_exceptions.Error as e:
            raise Error(e)

    def __enter__(self):
        if self._name:
            for cache_id, cache_class in six.iteritems(self._TYPES):
                if self._name.startswith(cache_id + '://'):
                    name = self._name[len(cache_id) + 3:]
                    if not name:
                        name = None
                    self._cache = self._OpenCache(cache_class, name)
                    return self._cache
        self._cache = self._OpenCache(resource_cache.ResourceCache, self._name)
        return self._cache

    def __exit__(self, typ, value, traceback):
        self._cache.Close(commit=typ is None)