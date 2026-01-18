import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
def test_subcommand_deferred_help(self):
    app, _ = make_app(deferred_help=True)
    with mock.patch.object(app, 'run_subcommand') as helper:
        app.run(['show', 'files', '--help'])
    helper.assert_called_once_with(['help', 'show', 'files'])