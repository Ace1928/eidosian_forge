from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesWorkloadSourcesOperationsGetRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesWorkloadSourcesOper
  ationsGetRequest object.

  Fields:
    name: The name of the operation resource.
  """
    name = _messages.StringField(1, required=True)