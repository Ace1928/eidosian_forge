from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.command_lib.privateca import exceptions
def CheckCreateCertificateAuthorityPermissions(project_ref, kms_key_ref=None):
    """Ensures that the current user has the required permissions to create a CA.

  Args:
    project_ref: The project where the new CA will be created.
    kms_key_ref: optional, The KMS key that will be used by the CA.

  Raises:
    InsufficientPermissionException: If the user is missing permissions.
  """
    _CheckAllPermissions(projects_api.TestIamPermissions(project_ref, _CA_CREATE_PERMISSIONS_ON_PROJECT).permissions, _CA_CREATE_PERMISSIONS_ON_PROJECT, 'project')
    if kms_key_ref:
        _CheckAllPermissions(kms_iam.TestCryptoKeyIamPermissions(kms_key_ref, _CA_CREATE_PERMISSIONS_ON_KEY).permissions, _CA_CREATE_PERMISSIONS_ON_KEY, 'KMS key')