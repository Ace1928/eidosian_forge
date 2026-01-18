from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
def test_other_unauthorized(self):
    """If the token is not at fault, a 401 error raises an exception."""
    SimulatedResponsesLaunchpad.responses = [Response(401, b'Some other error.')]
    self.assertRaises(Unauthorized, SimulatedResponsesLaunchpad.login_with, 'application name', authorization_engine=self.engine)