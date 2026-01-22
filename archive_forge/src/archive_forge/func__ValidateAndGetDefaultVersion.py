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
def _ValidateAndGetDefaultVersion(api_name, api_version):
    """Validates the API exists and gets the default version if not given."""
    api_name, _ = apis_internal._GetApiNameAndAlias(api_name)
    api_vers = apis_map.MAP.get(api_name, {})
    if not api_vers:
        raise UnknownAPIError(api_name)
    if api_version:
        if api_version not in api_vers:
            raise UnknownAPIVersionError(api_name, api_version)
        return api_version
    for version, api_def in six.iteritems(api_vers):
        if api_def.default_version:
            return version
    raise NoDefaultVersionError(api_name)