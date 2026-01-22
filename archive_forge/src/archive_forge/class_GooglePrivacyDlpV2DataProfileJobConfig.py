from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfileJobConfig(_messages.Message):
    """Configuration for setting up a job to scan resources for profile
  generation. Only one data profile configuration may exist per organization,
  folder, or project. The generated data profiles are retained according to
  the [data retention policy] (https://cloud.google.com/sensitive-data-
  protection/docs/data-profiles#retention).

  Fields:
    dataProfileActions: Actions to execute at the completion of the job.
    inspectTemplates: Detection logic for profile generation. Not all template
      features are used by profiles. FindingLimits, include_quote and
      exclude_info_types have no impact on data profiling. Multiple templates
      may be provided if there is data in multiple regions. At most one
      template must be specified per-region (including "global"). Each region
      is scanned using the applicable template. If no region-specific template
      is specified, but a "global" template is specified, it will be copied to
      that region and used instead. If no global or region-specific template
      is provided for a region with data, that region's data will not be
      scanned. For more information, see https://cloud.google.com/sensitive-
      data-protection/docs/data-profiles#data-residency.
    location: The data to scan.
    projectId: The project that will run the scan. The DLP service account
      that exists within this project must have access to all resources that
      are profiled, and the Cloud DLP API must be enabled.
  """
    dataProfileActions = _messages.MessageField('GooglePrivacyDlpV2DataProfileAction', 1, repeated=True)
    inspectTemplates = _messages.StringField(2, repeated=True)
    location = _messages.MessageField('GooglePrivacyDlpV2DataProfileLocation', 3)
    projectId = _messages.StringField(4)