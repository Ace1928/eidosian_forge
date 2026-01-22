from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportGenerationProgress(_messages.Message):
    """The ReportGenerationProgress is part of {google.longrunning.Operation}
  returned to client for every GET Operation request.

  Enums:
    StateValueValuesEnum: Output only. Highlights the current state of
      executation for report generation.

  Fields:
    destinationGcsBucket: Output only. The Cloud Storage bucket where the
      audit report will be uploaded once the evaluation process is completed.
    evaluationPercentComplete: Shows the progress of the CESS service
      evaluation process. The progress is defined in terms of percentage
      complete and is being fetched from the CESS service.
    failureReason: Output only. States the reason of failure during the audit
      report generation process. This field is set only if the state attribute
      is OPERATION_STATE_FAILED.
    reportGenerationPercentComplete: Shows the report generation progress of
      the CESS Result Processor Service. The // progress is defined in terms
      of percentage complete and is being fetched from the CESS service. If
      report_generation_in_progress is non zero then
      evaluation_percent_complete will be 100%.
    reportUploadingPercentComplete: Shows the report uploading progress of the
      CESS Result Processor Service. The progress is defined in terms of
      percentage complete and is being fetched from the CESS service. If
      report_uploading_in_progress is non zero then
      evaluation_percent_complete and report_generation_percent_complete will
      be 100%.
    state: Output only. Highlights the current state of executation for report
      generation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Highlights the current state of executation for report
    generation.

    Values:
      OPERATION_STATE_UNSPECIFIED: Unspecified. Invalid state.
      OPERATION_STATE_NOT_STARTED: Audit report generation process has not
        stated.
      OPERATION_STATE_EVALUATION_IN_PROGRESS: Audit Manager is currently
        evaluating the workloads against specific standard.
      OPERATION_STATE_EVALUATION_DONE: Audit Manager has completed Evaluation
        for the workload.
      OPERATION_STATE_EVIDENCE_REPORT_GENERATION_IN_PROGRESS: Audit Manager is
        creating audit report from the evaluated data.
      OPERATION_STATE_EVIDENCE_REPORT_GENERATION_DONE: Audit Manager has
        completed generation of the audit report.
      OPERATION_STATE_EVIDENCE_UPLOAD_IN_PROGRESS: Audit Manager is uploading
        the audit report and evidences to the customer provided destination.
      OPERATION_STATE_DONE: Audit report generation process is completed.
      OPERATION_STATE_FAILED: Audit report generation process has failed.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        OPERATION_STATE_NOT_STARTED = 1
        OPERATION_STATE_EVALUATION_IN_PROGRESS = 2
        OPERATION_STATE_EVALUATION_DONE = 3
        OPERATION_STATE_EVIDENCE_REPORT_GENERATION_IN_PROGRESS = 4
        OPERATION_STATE_EVIDENCE_REPORT_GENERATION_DONE = 5
        OPERATION_STATE_EVIDENCE_UPLOAD_IN_PROGRESS = 6
        OPERATION_STATE_DONE = 7
        OPERATION_STATE_FAILED = 8
    destinationGcsBucket = _messages.StringField(1)
    evaluationPercentComplete = _messages.FloatField(2)
    failureReason = _messages.StringField(3)
    reportGenerationPercentComplete = _messages.FloatField(4)
    reportUploadingPercentComplete = _messages.FloatField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)