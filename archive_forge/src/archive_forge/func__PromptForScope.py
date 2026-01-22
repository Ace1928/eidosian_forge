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
def _PromptForScope(self, ambiguous_names, attributes, services, resource_type, flag_names, prefix_filter):
    """Prompts user to specify a scope for ambiguous resources.

    Args:
      ambiguous_names: list(tuple(name, params, collection)),
        list of parameters which can be fed into resources.Parse.
      attributes: list(str), list of scopes to prompt over.
      services: list(apitool.base.py.base_api.BaseApiService), service for each
        attribute/scope.
      resource_type: str, collection name without api name.
      flag_names: list(str), flag names which can be used to specify scopes.
      prefix_filter: str, used to filter retrieved resources on backend.
    Returns:
      List of fully resolved names for provided ambiguous_names parameter.
    Raises:
      _InvalidPromptInvocation: if number of attributes does not match number of
        services.
    """

    def RaiseOnPromptFailure():
        """Call this to raise an exn when prompt cannot read from input stream."""
        phrases = ('one of ', 'flags') if len(flag_names) > 1 else ('', 'flag')
        raise compute_exceptions.FailedPromptError('Unable to prompt. Specify {0}the [{1}] {2}.'.format(phrases[0], ', '.join(flag_names), phrases[1]))
    if len(attributes) != len(services):
        raise _InvalidPromptInvocation()
    selected_resource_name = None
    selected_attribute = None
    if len(attributes) == 1:
        gce_suggestor = GCE_SUGGESTION_SOURCES.get(attributes[0]) or (lambda: None)
        gce_suggested_resource = gce_suggestor()
        if gce_suggested_resource:
            selected_attribute = attributes[0]
            selected_resource_name = self._PromptDidYouMeanScope(ambiguous_names, attributes[0], resource_type, gce_suggested_resource, RaiseOnPromptFailure)
    if selected_resource_name is None:
        choice_resources = {}
        for service, attribute in zip(services, attributes):
            choice_resources[attribute] = self.FetchChoiceResources(attribute, service, flag_names, prefix_filter)
        selected_attribute, selected_resource_name = self._PromptForScopeList(ambiguous_names, attributes, resource_type, choice_resources, RaiseOnPromptFailure)
    assert selected_resource_name is not None
    assert selected_attribute is not None
    result = []
    for ambigous_name, params, collection in ambiguous_names:
        new_params = params.copy()
        if selected_attribute in new_params:
            new_params[selected_attribute] = selected_resource_name
        try:
            resource_ref = self.resources.Parse(ambigous_name, params=new_params, collection=collection)
        except (resources.RequiredFieldOmittedException, properties.RequiredPropertyError):
            pass
        else:
            if hasattr(resource_ref, selected_attribute):
                result.append(resource_ref)
    return result