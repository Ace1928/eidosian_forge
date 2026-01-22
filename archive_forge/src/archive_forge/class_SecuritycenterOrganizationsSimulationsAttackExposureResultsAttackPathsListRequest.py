from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSimulationsAttackExposureResultsAttackPathsListRequest(_messages.Message):
    """A SecuritycenterOrganizationsSimulationsAttackExposureResultsAttackPaths
  ListRequest object.

  Fields:
    filter: The filter expression that filters the attack path in the
      response. Supported fields: * `valued_resources` supports =
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last `ListAttackPathsResponse`;
      indicates that this is a continuation of a prior `ListAttackPaths` call,
      and that the system should return the next page of data.
    parent: Required. Name of parent to list attack paths. Valid formats:
      "organizations/{organization}",
      "organizations/{organization}/simulations/{simulation}" "organizations/{
      organization}/simulations/{simulation}/attackExposureResults/{attack_exp
      osure_result_v2}" "organizations/{organization}/simulations/{simulation}
      /valuedResources/{valued_resource}"
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)