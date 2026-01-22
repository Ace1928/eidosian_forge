import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
class ChangelogParseError(Exception):
    """Indicates that the changelog could not be parsed"""
    is_user_error = True

    def __init__(self, line):
        self._line = line
        super(ChangelogParseError, self).__init__()

    def __str__(self):
        return 'Could not parse changelog: ' + self._line