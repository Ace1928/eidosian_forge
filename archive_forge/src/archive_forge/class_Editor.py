import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class Editor(MailClient):
    __doc__ = 'DIY mail client that uses commit message editor'
    supports_body = True

    def _get_merge_prompt(self, prompt, to, subject, attachment):
        """See MailClient._get_merge_prompt"""
        return '%s\n\nTo: %s\nSubject: %s\n\n%s' % (prompt, to, subject, attachment.decode('utf-8', 'replace'))

    def compose(self, prompt, to, subject, attachment, mime_subtype, extension, basename=None, body=None):
        """See MailClient.compose"""
        if not to:
            raise NoMailAddressSpecified()
        body = msgeditor.edit_commit_message(prompt, start_message=body)
        if body == '':
            raise NoMessageSupplied()
        email_message.EmailMessage.send(self.config, self.config.get('email'), to, subject, body, attachment, attachment_mime_subtype=mime_subtype)