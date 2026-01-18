import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '0'})
def test_retrieve_subject_token_not_allowd(self):
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(self.EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_ID_TOKEN).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE)
        with pytest.raises(ValueError) as excinfo:
            _ = credentials.retrieve_subject_token(None)
        assert excinfo.match('Executables need to be explicitly allowed')