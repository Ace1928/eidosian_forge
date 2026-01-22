from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmapihostingProjectsLocationsKrmApiHostsGetRequest(_messages.Message):
    """A KrmapihostingProjectsLocationsKrmApiHostsGetRequest object.

  Fields:
    name: Required. The name of this service resource in the format: 'projects
      /{project_id}/locations/{location}/krmApiHosts/{krm_api_host_id}'.
  """
    name = _messages.StringField(1, required=True)