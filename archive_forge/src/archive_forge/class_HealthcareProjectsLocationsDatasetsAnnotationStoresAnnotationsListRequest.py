from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsListRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsListRequest
  object.

  Enums:
    ViewValueValuesEnum: Controls which fields are populated in the response.

  Fields:
    filter: Restricts Annotations returned to those matching a filter.
      Functions available for filtering are: -
      `matches("annotation_source.cloud_healthcare_source.name", substring)`.
      Filter on `cloud_healthcare_source.name`. For example:
      `matches("annotation_source.cloud_healthcare_source.name", "some
      source")`. - `matches("annotation", substring)`. Filter on all fields of
      annotation. For example: `matches("annotation", "some-content")`. -
      `type("text")`, `type("image")`, `type("resource")`. Filter on the type
      of annotation `data`.
    pageSize: Limit on the number of Annotations to return in a single
      response. If not specified, 100 is used. May not be larger than 1000.
    pageToken: The next_page_token value returned from the previous List
      request, if any.
    parent: Required. Name of the Annotation store to retrieve Annotations
      from.
    view: Controls which fields are populated in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Controls which fields are populated in the response.

    Values:
      ANNOTATION_VIEW_UNSPECIFIED: Same as BASIC.
      ANNOTATION_VIEW_BASIC: Only `name`, `annotation_source` and
        `custom_data` fields are populated.
      ANNOTATION_VIEW_FULL: All fields are populated.
    """
        ANNOTATION_VIEW_UNSPECIFIED = 0
        ANNOTATION_VIEW_BASIC = 1
        ANNOTATION_VIEW_FULL = 2
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)