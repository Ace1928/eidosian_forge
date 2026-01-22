from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ExportRequest(_messages.Message):
    """Request body for [CreateExportRequest]

  Fields:
    csvDelimiter: Optional. Delimiter used in the CSV file, if `outputFormat`
      is set to `csv`. Defaults to the `,` (comma) character. Supported
      delimiter characters include comma (`,`), pipe (`|`), and tab (`\\t`).
    datastoreName: Required. Name of the preconfigured datastore.
    dateRange: Required. Date range of the data to export.
    description: Optional. Description of the export job.
    name: Required. Display name of the export job.
    outputFormat: Optional. Output format of the export. Valid values include:
      `csv` or `json`. Defaults to `json`. Note: Configure the delimiter for
      CSV output using the `csvDelimiter` property.
  """
    csvDelimiter = _messages.StringField(1)
    datastoreName = _messages.StringField(2)
    dateRange = _messages.MessageField('GoogleCloudApigeeV1DateRange', 3)
    description = _messages.StringField(4)
    name = _messages.StringField(5)
    outputFormat = _messages.StringField(6)