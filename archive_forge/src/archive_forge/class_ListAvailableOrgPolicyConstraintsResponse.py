from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAvailableOrgPolicyConstraintsResponse(_messages.Message):
    """The response returned from the `ListAvailableOrgPolicyConstraints`
  method. Returns all `Constraints` that could be set at this level of the
  hierarchy (contrast with the response from `ListPolicies`, which returns all
  policies which are set).

  Fields:
    constraints: The collection of constraints that are settable on the
      request resource.
    nextPageToken: Page token used to retrieve the next page. This is
      currently not used.
  """
    constraints = _messages.MessageField('Constraint', 1, repeated=True)
    nextPageToken = _messages.StringField(2)