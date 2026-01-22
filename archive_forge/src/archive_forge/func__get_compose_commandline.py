import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
def _get_compose_commandline(self, to, subject, attach_path, body=None, from_=None):
    """See ExternalMailClient._get_compose_commandline"""
    fd, self.temp_file = tempfile.mkstemp(prefix='bzr-send-', suffix='.scpt')
    try:
        os.write(fd, 'tell application "Mail"\n')
        os.write(fd, 'set newMessage to make new outgoing message\n')
        os.write(fd, 'tell newMessage\n')
        if to is not None:
            os.write(fd, 'make new to recipient with properties {address:"%s"}\n' % to)
        if from_ is not None:
            os.write(fd, 'set sender to "%s"\n' % from_.replace('"', '\\"'))
        if subject is not None:
            os.write(fd, 'set subject to "%s"\n' % subject.replace('"', '\\"'))
        if body is not None:
            os.write(fd, 'set content to "%s\\n\n"\n' % body.replace('"', '\\"').replace('\n', '\\n'))
        if attach_path is not None:
            os.write(fd, 'tell content to make new attachment with properties {file name:"%s"} at after the last paragraph\n' % self._encode_path(attach_path, 'attachment'))
        os.write(fd, 'set visible to true\n')
        os.write(fd, 'end tell\n')
        os.write(fd, 'end tell\n')
    finally:
        os.close(fd)
    return [self.temp_file]