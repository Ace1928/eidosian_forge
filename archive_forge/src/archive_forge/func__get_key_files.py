import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def _get_key_files(self, key_repo):
    key_files = dict()
    keys = dict()
    for filename in os.listdir(key_repo):
        path = os.path.join(key_repo, str(filename))
        if os.path.isfile(path):
            with open(path, 'r') as key_file:
                try:
                    key_id = int(filename)
                except ValueError:
                    pass
                else:
                    key = key_file.read()
                    if len(key) == 0:
                        LOG.warning('Ignoring empty key found in key repository: %s', path)
                        continue
                    key_files[key_id] = path
                    keys[key_id] = key
    return (key_files, keys)