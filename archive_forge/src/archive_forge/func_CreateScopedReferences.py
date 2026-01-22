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
def CreateScopedReferences(self, resource_names, scope_name, scope_arg, scope_service, resource_type, flag_names, prefix_filter=None):
    """Returns a list of resolved resource references for scoped resources."""
    resource_refs = []
    ambiguous_names = []
    resource_type = resource_type or self.resource_type
    collection = utils.GetApiCollection(resource_type)
    for resource_name in resource_names:
        params = {'project': properties.VALUES.core.project.GetOrFail, scope_name: scope_arg or getattr(properties.VALUES.compute, scope_name).GetOrFail}
        try:
            resource_ref = self.resources.Parse(resource_name, collection=collection, params=params)
        except (resources.RequiredFieldOmittedException, properties.RequiredPropertyError):
            ambiguous_names.append((resource_name, params, collection))
        else:
            resource_refs.append(resource_ref)
    if ambiguous_names and (not scope_arg):
        resource_refs += self._PromptForScope(ambiguous_names=ambiguous_names, attributes=[scope_name], services=[scope_service], resource_type=resource_type, flag_names=flag_names, prefix_filter=prefix_filter)
    return resource_refs