import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
class TestCheckaMetadataPath(object):

    def test_success(self):
        metadata_path = os.path.join(pytest.data_dir, 'context_aware_metadata.json')
        returned_path = _mtls_helper._check_dca_metadata_path(metadata_path)
        assert returned_path is not None

    def test_failure(self):
        metadata_path = os.path.join(pytest.data_dir, 'not_exists.json')
        returned_path = _mtls_helper._check_dca_metadata_path(metadata_path)
        assert returned_path is None