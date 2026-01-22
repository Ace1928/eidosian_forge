import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class XDGEmail(BodyExternalMailClient):
    __doc__ = 'xdg-email attempts to invoke the preferred mail client'
    _client_commands = ['xdg-email']

    def _get_compose_commandline(self, to, subject, attach_path, body=None):
        """See ExternalMailClient._get_compose_commandline"""
        if not to:
            raise NoMailAddressSpecified()
        commandline = [self._encode_safe(to)]
        if subject is not None:
            commandline.extend(['--subject', self._encode_safe(subject)])
        if attach_path is not None:
            commandline.extend(['--attach', self._encode_path(attach_path, 'attachment')])
        if body is not None:
            commandline.extend(['--body', self._encode_safe(body)])
        return commandline