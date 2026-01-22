from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseSecretManagerSecretVersion(secret_manager_version):
    """Parses a secret manager secret version name using configuration properties for fallback.

  Args:
    secret_manager_version: str, fully-qualified URL, or relative name

  Returns:
    str: the relative name of the secret version resource
  """
    return resources.REGISTRY.Parse(secret_manager_version, collection='secretmanager.projects.secrets.versions').RelativeName()