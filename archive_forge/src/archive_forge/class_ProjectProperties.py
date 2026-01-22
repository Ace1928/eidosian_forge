from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectProperties(_messages.Message):
    """A descriptor for defining project properties for a service. One service
  may have many consumer projects, and the service may want to behave
  differently depending on some properties on the project. For example, a
  project may be associated with a school, or a business, or a government
  agency, a business type property on the project may affect how a service
  responds to the client. This descriptor defines which properties are allowed
  to be set on a project.  Example:     project_properties:      properties:
  - name: NO_WATERMARK        type: BOOL        description: Allows usage of
  the API without watermarks.      - name: EXTENDED_TILE_CACHE_PERIOD
  type: INT64

  Fields:
    properties: List of per consumer project-specific properties.
  """
    properties = _messages.MessageField('Property', 1, repeated=True)