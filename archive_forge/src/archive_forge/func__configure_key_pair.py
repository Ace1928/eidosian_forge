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
def _configure_key_pair(config, compute):
    """Configure SSH access, using an existing key pair if possible.

    Creates a project-wide ssh key that can be used to access all the instances
    unless explicitly prohibited by instance config.

    The ssh-keys created by ray are of format:

      [USERNAME]:ssh-rsa [KEY_VALUE] [USERNAME]

    where:

      [USERNAME] is the user for the SSH key, specified in the config.
      [KEY_VALUE] is the public SSH key value.
    """
    config = copy.deepcopy(config)
    if 'ssh_private_key' in config['auth']:
        return config
    ssh_user = config['auth']['ssh_user']
    project = compute.projects().get(project=config['provider']['project_id']).execute()
    ssh_keys_str = next((item for item in project['commonInstanceMetadata'].get('items', []) if item['key'] == 'ssh-keys'), {}).get('value', '')
    ssh_keys = ssh_keys_str.split('\n') if ssh_keys_str else []
    key_found = False
    for i in range(10):
        key_name = key_pair_name(i, config['provider']['region'], config['provider']['project_id'], ssh_user)
        public_key_path, private_key_path = key_pair_paths(key_name)
        for ssh_key in ssh_keys:
            key_parts = ssh_key.split(' ')
            if len(key_parts) != 3:
                continue
            if key_parts[2] == ssh_user and os.path.exists(private_key_path):
                key_found = True
                break
        os.makedirs(os.path.expanduser('~/.ssh'), exist_ok=True)
        if not key_found and (not os.path.exists(private_key_path)):
            logger.info('_configure_key_pair: Creating new key pair {}'.format(key_name))
            public_key, private_key = generate_rsa_key_pair()
            _create_project_ssh_key_pair(project, public_key, ssh_user, compute)
            private_key_dir = os.path.dirname(private_key_path)
            os.makedirs(private_key_dir, exist_ok=True)
            with open(private_key_path, 'w', opener=partial(os.open, mode=384)) as f:
                f.write(private_key)
            with open(public_key_path, 'w') as f:
                f.write(public_key)
            key_found = True
            break
        if key_found:
            break
    assert key_found, 'SSH keypair for user {} not found for {}'.format(ssh_user, private_key_path)
    assert os.path.exists(private_key_path), 'Private key file {} not found for user {}'.format(private_key_path, ssh_user)
    logger.info('_configure_key_pair: Private key not specified in config, using{}'.format(private_key_path))
    config['auth']['ssh_private_key'] = private_key_path
    return config