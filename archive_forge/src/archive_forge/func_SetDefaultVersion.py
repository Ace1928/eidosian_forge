from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
def SetDefaultVersion(api_name, api_version):
    """Resets default version for given api."""
    api_def = apis_internal.GetApiDef(api_name, api_version)
    default_version = apis_internal._GetDefaultVersion(api_name)
    default_api_def = apis_internal.GetApiDef(api_name, default_version)
    default_api_def.default_version = False
    api_def.default_version = True