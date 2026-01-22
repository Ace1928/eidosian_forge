import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class Mutt(BodyExternalMailClient):
    __doc__ = 'Mutt mail client.'
    _client_commands = ['mutt']

    def _get_compose_commandline(self, to, subject, attach_path, body=None):
        """See ExternalMailClient._get_compose_commandline"""
        message_options = []
        if subject is not None:
            message_options.extend(['-s', self._encode_safe(subject)])
        if attach_path is not None:
            message_options.extend(['-a', self._encode_path(attach_path, 'attachment')])
        if body is not None:
            self._temp_file = tempfile.NamedTemporaryFile(prefix='mutt-body-', suffix='.txt', mode='w+')
            self._temp_file.write(body)
            self._temp_file.flush()
            message_options.extend(['-i', self._temp_file.name])
        if to is not None:
            message_options.extend(['--', self._encode_safe(to)])
        return message_options