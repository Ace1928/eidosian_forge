import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_executable_fail(self):
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=None, returncode=1)):
        credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            _ = credentials.retrieve_subject_token(None)
        assert excinfo.match('Executable exited with non-zero return code 1. Error: None')