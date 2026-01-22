from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerScansListRequest(_messages.Message):
    """A SpannerScansListRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which parts of the Scan should be returned
      in the response. Note, only the SUMMARY view (the default) is currently
      supported for ListScans.

  Fields:
    filter: A filter expression to restrict the results based on information
      present in the available Scan collection. The filter applies to all
      fields within the Scan message except for `data`.
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The unique name of the parent resource, specific to the
      Database service implementing this interface.
    view: Specifies which parts of the Scan should be returned in the
      response. Note, only the SUMMARY view (the default) is currently
      supported for ListScans.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which parts of the Scan should be returned in the response.
    Note, only the SUMMARY view (the default) is currently supported for
    ListScans.

    Values:
      VIEW_UNSPECIFIED: Not specified, equivalent to SUMMARY.
      SUMMARY: Server responses only include `name`, `details`, `start_time`
        and `end_time`. The default value. Note, the ListScans method may only
        use this view type, others view types are not supported.
      FULL: Full representation of the scan is returned in the server
        response, including `data`.
    """
        VIEW_UNSPECIFIED = 0
        SUMMARY = 1
        FULL = 2
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)