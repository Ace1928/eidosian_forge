from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
def GetParserForCollection(self, collection, api_version=None):
    """Returns a parser object for collection.

    Args:
      collection: str, The resource collection name.
      api_version: str, The API version, None for the default version.

    Raises:
      InvalidCollectionException: If there is no parser.

    Returns:
      The parser object for collection.
    """
    api_name = _APINameFromCollection(collection)
    api_version = self.RegisterApiByName(api_name, api_version=api_version)
    parser = self.parsers_by_collection.get(api_name, {}).get(api_version, {}).get(collection, None)
    if parser is None:
        raise InvalidCollectionException(collection, api_version)
    return parser