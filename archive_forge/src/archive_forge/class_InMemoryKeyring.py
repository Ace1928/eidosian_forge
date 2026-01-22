from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class InMemoryKeyring:
    """A keyring that saves passwords only in memory."""

    def __init__(self):
        self.data = {}

    def set_password(self, service, username, password):
        self.data[service, username] = password

    def get_password(self, service, username):
        return self.data.get((service, username))