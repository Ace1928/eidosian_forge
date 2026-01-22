from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
def BuildSecrets(project_name, secret_list, namespace, client=None):
    """Fetch secrets from Secret Manager and create k8s secrets with the data."""
    if client is None:
        client = _SecretsClient()
    secrets = []
    for secret in secret_list:
        secrets.append(_BuildSecret(client, project_name, secret.name, secret.mapped_secret, secret.versions, namespace))
    return secrets