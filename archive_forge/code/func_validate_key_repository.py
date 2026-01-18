import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def validate_key_repository(self, requires_write=False):
    """Validate permissions on the key repository directory."""
    is_valid = os.access(self.key_repository, os.R_OK) and os.access(self.key_repository, os.X_OK)
    if requires_write:
        is_valid = is_valid and os.access(self.key_repository, os.W_OK)
    if not is_valid:
        LOG.error('Either [%(config_group)s] key_repository does not exist or Keystone does not have sufficient permission to access it: %(key_repo)s', {'key_repo': self.key_repository, 'config_group': self.config_group})
    else:
        stat_info = os.stat(self.key_repository)
        if stat_info.st_mode & stat.S_IROTH or stat_info.st_mode & stat.S_IXOTH:
            LOG.warning('key_repository is world readable: %s', self.key_repository)
    return is_valid