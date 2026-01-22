import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class MAPIClient(BodyExternalMailClient):
    __doc__ = 'Default Windows mail client launched using MAPI.'

    def _compose(self, prompt, to, subject, attach_path, mime_subtype, extension, body=None):
        """See ExternalMailClient._compose.

        This implementation uses MAPI via the simplemapi ctypes wrapper
        """
        from .util import simplemapi
        try:
            simplemapi.SendMail(to or '', subject or '', body or '', attach_path)
        except simplemapi.MAPIError as e:
            if e.code != simplemapi.MAPI_USER_ABORT:
                raise MailClientNotFound(['MAPI supported mail client (error %d)' % (e.code,)])