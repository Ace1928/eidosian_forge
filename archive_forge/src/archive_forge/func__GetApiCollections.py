from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.generated_clients.apis import apis_map
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.parse import urlparse
def _GetApiCollections(api_name, api_version):
    """Yields all collections for for given api."""
    try:
        resources_module = _GetResourceModule(api_name, api_version)
    except ImportError:
        pass
    else:
        for collection in resources_module.Collections:
            yield resource_util.CollectionInfo(api_name, api_version, resources_module.BASE_URL, resources_module.DOCS_URL, collection.collection_name, collection.path, collection.flat_paths, collection.params, collection.enable_uri_parsing)