import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
def _enforce_microversion(self):
    if self._api.microversion == '1.0':
        raise NotImplementedError('Server does not support secret consumers.  Minimum key-manager microversion required: 1.1')