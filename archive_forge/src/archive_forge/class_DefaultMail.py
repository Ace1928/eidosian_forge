import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class DefaultMail(MailClient):
    __doc__ = 'Default mail handling.  Tries XDGEmail (or MAPIClient on Windows),\n    falls back to Editor'
    supports_body = True

    def _mail_client(self):
        """Determine the preferred mail client for this platform"""
        if osutils.supports_mapi():
            return MAPIClient(self.config)
        else:
            return XDGEmail(self.config)

    def compose(self, prompt, to, subject, attachment, mime_subtype, extension, basename=None, body=None):
        """See MailClient.compose"""
        try:
            return self._mail_client().compose(prompt, to, subject, attachment, mime_subtype, extension, basename, body)
        except MailClientNotFound:
            return Editor(self.config).compose(prompt, to, subject, attachment, mime_subtype, extension, body)

    def compose_merge_request(self, to, subject, directive, basename=None, body=None):
        """See MailClient.compose_merge_request"""
        try:
            return self._mail_client().compose_merge_request(to, subject, directive, basename=basename, body=body)
        except MailClientNotFound:
            return Editor(self.config).compose_merge_request(to, subject, directive, basename=basename, body=body)