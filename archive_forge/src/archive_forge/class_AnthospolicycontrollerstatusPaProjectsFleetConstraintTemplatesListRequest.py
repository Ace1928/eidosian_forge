from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesListRequest(_messages.Message):
    """A
  AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesListRequest
  object.

  Fields:
    pageSize: The maximum number of fleet constraint templates to return. The
      service may return fewer than this value. If unspecified or 0, defaults
      to 500 results. The maximum value is 2000; values above 2000 will be
      coerced to 2000.
    pageToken: A page token, received from a previous
      ListFleetConstraintTemplates call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      ListFleetConstraintTemplates must match the call that provided the page
      token.
    parent: Required. The project id for which to fetch fleet constraint
      templates. Format: projects/{project_id}.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)