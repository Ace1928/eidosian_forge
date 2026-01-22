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
def _GetResourceModule(api_name, api_version):
    """Imports and returns given api resources module."""
    api_def = GetApiDef(api_name, api_version)
    if api_def.apitools:
        return __import__(api_def.apitools.class_path + '.' + 'resources', fromlist=['something'])
    return __import__(api_def.gapic.class_path + '.' + 'resources', fromlist=['something'])