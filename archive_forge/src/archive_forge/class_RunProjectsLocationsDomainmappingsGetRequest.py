from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsDomainmappingsGetRequest(_messages.Message):
    """A RunProjectsLocationsDomainmappingsGetRequest object.

  Fields:
    name: Required. The name of the domain mapping to retrieve. For Cloud Run
      (fully managed), replace {namespace} with the project ID or number. It
      takes the form namespaces/{namespace}. For example:
      namespaces/PROJECT_ID
  """
    name = _messages.StringField(1, required=True)