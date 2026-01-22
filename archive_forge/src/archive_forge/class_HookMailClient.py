import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class HookMailClient(mail_client.MailClient):
    """Mail client for testing hooks."""

    def __init__(self, config):
        self.body = None
        self.config = config

    def compose(self, prompt, to, subject, attachment, mime_subtype, extension, basename=None, body=None):
        self.body = body