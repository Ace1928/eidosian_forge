from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import unittest
from unittest import mock
from google.auth import exceptions as google_auth_exceptions
from gslib.command_runner import CommandRunner
from gslib.utils import system_util
import gslib
import gslib.tests.testcase as testcase
@mock.patch.object(system_util, 'InvokedViaCloudSdk', autospec=True)
@mock.patch.object(gslib.__main__, '_OutputAndExit', autospec=True)
def test_translates_oauth_error_cloudsdk(self, mock_output_and_exit, mock_invoke_via_cloud_sdk):
    mock_invoke_via_cloud_sdk.return_value = True
    command_runner = CommandRunner()
    with mock.patch.object(command_runner, 'RunNamedCommand') as mock_run:
        fake_error = google_auth_exceptions.OAuthError('fake error message')
        mock_run.side_effect = fake_error
        gslib.__main__._RunNamedCommandAndHandleExceptions(command_runner, command_name='fake')
        mock_output_and_exit.assert_called_once_with('Your credentials are invalid. Please run\n$ gcloud auth login', fake_error)