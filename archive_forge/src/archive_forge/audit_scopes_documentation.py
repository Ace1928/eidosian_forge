from googlecloudsdk.api_lib.audit_manager import util
Generate an Audit Scope.

    Args:
      scope: str, the scope for which to generate the scope.
      compliance_standard: str, Compliance standard against which the scope
        must be generated.
      report_format: str, The format in which the audit scope should be
        generated.
      is_parent_folder: bool, whether the parent is folder and not project.

    Returns:
      Described audit scope resource.
    