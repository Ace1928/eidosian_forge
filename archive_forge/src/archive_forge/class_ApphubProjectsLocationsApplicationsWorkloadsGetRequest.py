from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsApplicationsWorkloadsGetRequest(_messages.Message):
    """A ApphubProjectsLocationsApplicationsWorkloadsGetRequest object.

  Fields:
    name: Required. Fully qualified name of the Workload to fetch. Expected
      format: `projects/{project}/locations/{location}/applications/{applicati
      on}/workloads/{workload}`.
  """
    name = _messages.StringField(1, required=True)