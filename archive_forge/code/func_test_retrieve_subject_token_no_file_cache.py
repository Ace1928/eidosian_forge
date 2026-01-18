import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_no_file_cache(self):
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE = {'command': 'command', 'timeout_millis': 30000}
    ACTUAL_CREDENTIAL_SOURCE = {'executable': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE}
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(self.EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_ID_TOKEN).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=ACTUAL_CREDENTIAL_SOURCE)
        subject_token = credentials.retrieve_subject_token(None)
        assert subject_token == self.EXECUTABLE_OIDC_TOKEN