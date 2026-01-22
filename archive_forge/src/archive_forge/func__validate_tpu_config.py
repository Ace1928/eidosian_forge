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
def _validate_tpu_config(node: dict):
    """Validate the provided node with TPU support.

    If the config is malformed, users will run into an error but this function
    will raise the error at config parsing time. This only tests very simple assertions.

    Raises: `ValueError` in case the input is malformed.

    """
    if 'acceleratorType' in node and 'acceleratorConfig' in node:
        raise ValueError('For TPU usage, acceleratorType and acceleratorConfig cannot both be set.')
    if 'acceleratorType' in node:
        accelerator_type = node['acceleratorType']
        if not TPUAcceleratorManager.is_valid_tpu_accelerator_type(accelerator_type):
            raise ValueError(f'`acceleratorType` should match v(generation)-(cores/chips). Got {accelerator_type}.')
    else:
        accelerator_config = node['acceleratorConfig']
        if 'type' not in accelerator_config or 'topology' not in accelerator_config:
            raise ValueError(f"acceleratorConfig expects 'type' and 'topology'. Got {accelerator_config}")
        generation = node['acceleratorConfig']['type']
        topology = node['acceleratorConfig']['topology']
        generation_pattern = re.compile('^V\\d+[a-zA-Z]*$')
        topology_pattern = re.compile('^\\d+x\\d+(x\\d+)?$')
        if not generation_pattern.match(generation):
            raise ValueError(f'type should match V(generation). Got {generation}.')
        if generation == 'V2' or generation == 'V3':
            raise ValueError(f'acceleratorConfig is not supported on V2/V3 TPUs. Got {generation}.')
        if not topology_pattern.match(topology):
            raise ValueError(f'topology should be of form axbxc or axb. Got {topology}.')