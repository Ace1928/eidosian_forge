from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def PromptForMultiScopedReferences(self, resource_names, scope_names, scope_services, resource_types, flag_names):
    """Prompt for resources, which can be placed in several different scopes."""
    if len(scope_names) != len(scope_services) or len(scope_names) != len(resource_types):
        raise _InvalidPromptInvocation()
    resource_refs = []
    ambiguous_names = []
    for resource_name in resource_names:
        for resource_type in resource_types:
            collection = utils.GetApiCollection(resource_type)
            params = {'project': properties.VALUES.core.project.GetOrFail}
            collection_info = self.resources.GetCollectionInfo(collection)
            if 'zone' in collection_info.params:
                params['zone'] = properties.VALUES.compute.zone.GetOrFail
            elif 'region' in collection_info.params:
                params['region'] = properties.VALUES.compute.region.GetOrFail
            try:
                resource_ref = self.resources.Parse(resource_name, params=params, collection=collection)
            except resources.WrongResourceCollectionException:
                pass
            except (resources.RequiredFieldOmittedException, properties.RequiredPropertyError):
                ambiguous_names.append((resource_name, params, collection))
            else:
                resource_refs.append(resource_ref)
    if ambiguous_names:
        resource_refs += self._PromptForScope(ambiguous_names=ambiguous_names, attributes=scope_names, services=scope_services, resource_type=resource_types[0], flag_names=flag_names, prefix_filter=None)
    return resource_refs