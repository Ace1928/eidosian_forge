import copy
import json
import logging
import os
import re
import time
from functools import partial, reduce
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from googleapiclient import discovery, errors
from ray._private.accelerators import TPUAcceleratorManager, tpu
from ray.autoscaler._private.gcp.node import MAX_POLLS, POLL_INTERVAL, GCPNodeType
from ray.autoscaler._private.util import check_legacy_fields
def _add_iam_policy_binding(service_account, roles, crm):
    """Add new IAM roles for the service account."""
    project_id = service_account['projectId']
    email = service_account['email']
    member_id = 'serviceAccount:' + email
    policy = crm.projects().getIamPolicy(resource=project_id, body={}).execute()
    already_configured = True
    for role in roles:
        role_exists = False
        for binding in policy['bindings']:
            if binding['role'] == role:
                if member_id not in binding['members']:
                    binding['members'].append(member_id)
                    already_configured = False
                role_exists = True
        if not role_exists:
            already_configured = False
            policy['bindings'].append({'members': [member_id], 'role': role})
    if already_configured:
        return
    result = crm.projects().setIamPolicy(resource=project_id, body={'policy': policy}).execute()
    return result