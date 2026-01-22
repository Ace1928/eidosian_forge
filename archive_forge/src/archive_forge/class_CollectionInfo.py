from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
class CollectionInfo(object):
    """Holds information about a resource collection.

  Attributes:
      api_name: str, name of the api of resources parsed by this parser.
      api_version: str, version id for this api.
      path: str, Atomic URI template for this resource.
      flat_paths: {name->path}, Named detailed URI templates for this resource.
        If there is an entry ''->path it replaces path and corresponding param
        attributes for resources parsing. path and params are not used in this
        case. Also note that key in this dictionary is referred as
        subcollection, as it extends 'name' attribute.
      params: list(str), description of parameters in the path.
      name: str, collection name for this resource without leading api_name.
      base_url: str, URL for service providing these resources.
      docs_url: str, URL to the API reference docs for this API.
      enable_uri_parsing: bool, whether to register a parser to build up a
        search tree to match URLs against URL templates.
  """

    def __init__(self, api_name, api_version, base_url, docs_url, name, path, flat_paths, params, enable_uri_parsing=True):
        self.api_name = api_name
        self.api_version = api_version
        self.base_url = base_url
        self.docs_url = docs_url
        self.name = name
        self.path = path
        self.flat_paths = flat_paths
        self.params = params
        self.enable_uri_parsing = enable_uri_parsing

    @property
    def full_name(self):
        return self.api_name + '.' + self.name

    def GetSubcollection(self, collection_name):
        name = self.full_name
        if collection_name.startswith(name):
            return collection_name[len(name) + 1:]
        raise KeyError('{0} does not exist in {1}'.format(collection_name, name))

    def GetPathRegEx(self, subcollection):
        """Returns regex for matching path template."""
        path = self.GetPath(subcollection)
        parts = []
        prev_end = 0
        for match in re.finditer('({[^}]+}/)|({[^}]+})$', path):
            parts.append(path[prev_end:match.start()])
            parts.append('([^/]+)')
            if match.group(1):
                parts.append('/')
            prev_end = match.end()
        if prev_end == len(path):
            parts[-1] = '(.*)$'
        return ''.join(parts)

    def GetParams(self, subcollection):
        """Returns ordered list of parameters for given subcollection.

    Args:
      subcollection: str, key name for flat_paths. If self.flat_paths is empty
        use '' (or any other falsy value) for subcollection to get default path
        parameters.

    Returns:
      Paramaters present in specified subcollection path.
    Raises:
      KeyError if given subcollection does not exists.
    """
        if not subcollection and (not self.flat_paths):
            return self.params
        return GetParamsFromPath(self.flat_paths[subcollection])

    def GetPath(self, subcollection):
        """Returns uri template path for given subcollection."""
        if not subcollection and (not self.flat_paths):
            return self.path
        return self.flat_paths[subcollection]

    def __eq__(self, other):
        return self.api_name == other.api_name and self.api_version == other.api_version and (self.name == other.name)

    def __ne__(self, other):
        return not self == other

    @classmethod
    def _CmpHelper(cls, x, y):
        """Just a helper equivalent to the cmp() function in Python 2."""
        return (x > y) - (x < y)

    def __lt__(self, other):
        return self._CmpHelper((self.api_name, self.api_version, self.name), (other.api_name, other.api_version, other.name)) < 0

    def __gt__(self, other):
        return self._CmpHelper((self.api_name, self.api_version, self.name), (other.api_name, other.api_version, other.name)) > 0

    def __le__(self, other):
        return not self.__gt__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __str__(self):
        return self.full_name