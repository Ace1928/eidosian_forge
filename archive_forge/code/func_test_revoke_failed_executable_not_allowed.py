import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '0'})
def test_revoke_failed_executable_not_allowed(self):
    credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE, interactive=True)
    with pytest.raises(ValueError) as excinfo:
        _ = credentials.revoke(None)
    assert excinfo.match('Executables need to be explicitly allowed')