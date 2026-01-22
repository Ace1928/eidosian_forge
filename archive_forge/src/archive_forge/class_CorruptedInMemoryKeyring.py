import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class CorruptedInMemoryKeyring(InMemoryKeyring):

    def get_password(self, service, username):
        return 'bad'