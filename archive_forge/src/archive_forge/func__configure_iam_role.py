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
def _configure_iam_role(config, crm, iam):
    """Setup a gcp service account with IAM roles.

    Creates a gcp service acconut and binds IAM roles which allow it to control
    control storage/compute services. Specifically, the head node needs to have
    an IAM role that allows it to create further gce instances and store items
    in google cloud storage.

    TODO: Allow the name/id of the service account to be configured
    """
    config = copy.deepcopy(config)
    email = SERVICE_ACCOUNT_EMAIL_TEMPLATE.format(account_id=DEFAULT_SERVICE_ACCOUNT_ID, project_id=config['provider']['project_id'])
    service_account = _get_service_account(email, config, iam)
    if service_account is None:
        logger.info('_configure_iam_role: Creating new service account {}'.format(DEFAULT_SERVICE_ACCOUNT_ID))
        service_account = _create_service_account(DEFAULT_SERVICE_ACCOUNT_ID, DEFAULT_SERVICE_ACCOUNT_CONFIG, config, iam)
    assert service_account is not None, 'Failed to create service account'
    if config['provider'].get(HAS_TPU_PROVIDER_FIELD, False):
        roles = DEFAULT_SERVICE_ACCOUNT_ROLES + TPU_SERVICE_ACCOUNT_ROLES
    else:
        roles = DEFAULT_SERVICE_ACCOUNT_ROLES
    _add_iam_policy_binding(service_account, roles, crm)
    config['head_node']['serviceAccounts'] = [{'email': service_account['email'], 'scopes': ['https://www.googleapis.com/auth/cloud-platform']}]
    return config