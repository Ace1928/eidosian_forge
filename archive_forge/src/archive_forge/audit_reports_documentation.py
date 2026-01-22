from googlecloudsdk.api_lib.audit_manager import util
Generate an Audit Report.

    Args:
      scope: str, the scope for which to generate the report.
      compliance_standard: str, Compliance standard against which the Report
        must be generated.
      report_format: str, The format in which the audit report should be
        generated.
      gcs_uri: str, Destination Cloud storage bucket where report and evidence
        must be uploaded.
      is_parent_folder: bool, whether the parent is folder and not project.

    Returns:
      Described audit operation resource.
    