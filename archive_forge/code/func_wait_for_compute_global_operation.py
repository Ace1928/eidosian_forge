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
def wait_for_compute_global_operation(project_name, operation, compute):
    """Poll for global compute operation until finished."""
    logger.info('wait_for_compute_global_operation: Waiting for operation {} to finish...'.format(operation['name']))
    for _ in range(MAX_POLLS):
        result = compute.globalOperations().get(project=project_name, operation=operation['name']).execute()
        if 'error' in result:
            raise Exception(result['error'])
        if result['status'] == 'DONE':
            logger.info('wait_for_compute_global_operation: Operation done.')
            break
        time.sleep(POLL_INTERVAL)
    return result