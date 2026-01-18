import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1', 'GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE': '1'})
def test_retrieve_subject_token_failed_interactive_mode(self, tmpdir):
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE = tmpdir.join('actual_output_file')
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE = {'command': 'command', 'interactive_timeout_millis': 300000, 'output_file': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}
    ACTUAL_CREDENTIAL_SOURCE = {'executable': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE}
    with open(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        json.dump(self.EXECUTABLE_FAILED_RESPONSE, output_file)
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], returncode=0)):
        credentials = self.make_pluggable(audience=WORKFORCE_AUDIENCE, credential_source=ACTUAL_CREDENTIAL_SOURCE, interactive=True)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            _ = credentials.retrieve_subject_token(None)
        assert excinfo.match('Executable returned unsuccessful response: code: 401, message: Permission denied. Caller not authorized.')
        os.remove(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE)