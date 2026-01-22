from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def _RequestCollection(self):
    """Gets the collection that matches the API parameters of this method.

    Methods apply to elements of a collection. The resource argument is always
    of the type of that collection.  List is an exception where you are listing
    items of that collection so the argument to be provided is that of the
    parent collection. This method returns the collection that should be used
    to parse the resource for this specific method.

    Returns:
      APICollection, The collection to use or None if no parent collection could
      be found.
    """
    if self.detailed_params == self.collection.detailed_params:
        return self.collection
    collections = GetAPICollections(self.collection.api_name, self.collection.api_version)
    for c in collections:
        if self.detailed_params == c.detailed_params:
            return c
    return None