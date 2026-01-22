from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
def _BuildSecret(client, project, secret_name, mapped_secret, versions, namespace):
    """Build the k8s secret resource for minikube from Secret Manager data."""
    if secrets_mapping.SpecialVersion.MOUNT_ALL in versions:
        raise ValueError('local development requires you to specify all secret versions that you need to use.')
    secrets = {}
    for version in versions:
        secrets[version] = client.GetSecretData(project, secret_name, mapped_secret, version)
    return _BuildK8sSecret(secret_name, secrets, namespace)