from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import resources
import six
def Resource(collection, api_version=None):
    """A hook to do basic parsing of a resource in a single flag.

  Args:
    collection: str, The collection the resource is in.
    api_version: str, An optional version to use to parse this resource.

  Returns:
    f(value) -> resource_ref, An argument processing function that returns the
    parsed resource reference.
  """
    collection_info = resources.REGISTRY.GetCollectionInfo(collection, api_version=api_version)
    params = collection_info.GetParams('')

    def Parse(value):
        if not value:
            return None
        ref = resources.REGISTRY.Parse(value, collection=collection, params={k: f for k, f in six.iteritems(arg_utils.DEFAULT_PARAMS) if k in params})
        return ref
    return Parse