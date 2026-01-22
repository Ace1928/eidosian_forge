import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def _test_returncode_helper(self, returncode, expected):
    with mock.patch.object(wrapper, 'start_subprocess') as mock_start:
        with mock.patch('sys.exit') as mock_exit:
            mock_start.return_value.wait.return_value = returncode
            cmd.run_one_command(None, mock.Mock(), None, None)
    mock_exit.assert_called_once_with(expected)