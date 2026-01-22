def ConvertFirewallPolicy(policy, project):
    """Converts Firewall Policy to terraform script.

  Args:
    policy: Network Firewall Policy
    project: Project container of Firewall Policy

  Returns:
    Terraform script
  """
    return 'resource "google_compute_network_firewall_policy" "auto_generated_firewall_policy" {\n  name = "%s"\n  project = "%s"\n  description = "%s"\n}\n' % (policy.name, project, policy.description)