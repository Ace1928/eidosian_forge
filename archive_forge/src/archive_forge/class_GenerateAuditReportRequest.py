from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateAuditReportRequest(_messages.Message):
    """Message for requesting the Audit Report.

  Enums:
    ReportFormatValueValuesEnum: Required. The format in which the audit
      report should be created.

  Fields:
    complianceStandard: Required. Compliance Standard against which the Scope
      Report must be generated. Eg: FEDRAMP_MODERATE
    gcsUri: Destination Cloud storage bucket where report and evidence must be
      uploaded. The Cloud storage bucket provided here must be selected among
      the buckets entered during the enrollment process.
    reportFormat: Required. The format in which the audit report should be
      created.
  """

    class ReportFormatValueValuesEnum(_messages.Enum):
        """Required. The format in which the audit report should be created.

    Values:
      AUDIT_REPORT_FORMAT_UNSPECIFIED: Unspecified. Invalid state.
      AUDIT_REPORT_FORMAT_ODF: Audit Report creation format is Open Document.
    """
        AUDIT_REPORT_FORMAT_UNSPECIFIED = 0
        AUDIT_REPORT_FORMAT_ODF = 1
    complianceStandard = _messages.StringField(1)
    gcsUri = _messages.StringField(2)
    reportFormat = _messages.EnumField('ReportFormatValueValuesEnum', 3)