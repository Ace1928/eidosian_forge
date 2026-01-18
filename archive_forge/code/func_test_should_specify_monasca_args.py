from unittest import mock
from oslotest import base
from monascaclient import shell
def test_should_specify_monasca_args(self):
    expected_args = ['--monasca-api-url', '--monasca-api-version', '--monasca_api_url', '--monasca_api_version']
    parser = mock.Mock()
    parser.add_argument = aa = mock.Mock()
    shell.MonascaShell._append_monasca_args(parser)
    aa.assert_called()
    for mc in aa.mock_calls:
        name = mc[1][0]
        self.assertIn(name, expected_args)