from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsAnalyticsExportsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsAnalyticsExportsGetRequest object.

  Fields:
    name: Required. Resource name of the export to get.
  """
    name = _messages.StringField(1, required=True)