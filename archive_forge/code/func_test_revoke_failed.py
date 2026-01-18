import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_revoke_failed(self):
    testData = {'non_interactive_mode': {'interactive': False, 'expectErrType': ValueError, 'expectErrPattern': 'Revoke is only enabled under interactive mode.'}, 'executable_failed': {'returncode': 1, 'expectErrType': exceptions.RefreshError, 'expectErrPattern': 'Auth revoke failed on executable.'}, 'response_validation_missing_version': {'response': {}, 'expectErrType': ValueError, 'expectErrPattern': 'The executable response is missing the version field.'}, 'response_validation_invalid_version': {'response': {'version': 2}, 'expectErrType': exceptions.RefreshError, 'expectErrPattern': 'Executable returned unsupported version.'}, 'response_validation_missing_success': {'response': {'version': 1}, 'expectErrType': ValueError, 'expectErrPattern': 'The executable response is missing the success field.'}, 'response_validation_failed_with_success_field_is_false': {'response': {'version': 1, 'success': False}, 'expectErrType': exceptions.RefreshError, 'expectErrPattern': 'Revoke failed with unsuccessful response.'}}
    for data in testData.values():
        with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(data.get('response')).encode('UTF-8'), returncode=data.get('returncode', 0))):
            credentials = self.make_pluggable(audience=WORKFORCE_AUDIENCE, service_account_impersonation_url=SERVICE_ACCOUNT_IMPERSONATION_URL, credential_source=self.CREDENTIAL_SOURCE, interactive=data.get('interactive', True))
            with pytest.raises(data.get('expectErrType')) as excinfo:
                _ = credentials.revoke(None)
            assert excinfo.match(data.get('expectErrPattern'))