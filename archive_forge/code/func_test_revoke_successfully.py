import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_revoke_successfully(self):
    ACTUAL_RESPONSE = {'version': 1, 'success': True}
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(ACTUAL_RESPONSE).encode('utf-8'), returncode=0)):
        credentials = self.make_pluggable(audience=WORKFORCE_AUDIENCE, credential_source=self.CREDENTIAL_SOURCE, interactive=True)
        _ = credentials.revoke(None)