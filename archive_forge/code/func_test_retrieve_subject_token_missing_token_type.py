import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_missing_token_type(self):
    EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE = {'version': 1, 'success': True, 'id_token': self.EXECUTABLE_OIDC_TOKEN, 'expiration_time': 9999999999}
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE)
        with pytest.raises(ValueError) as excinfo:
            _ = credentials.retrieve_subject_token(None)
        assert excinfo.match('The executable response is missing the token_type field.')