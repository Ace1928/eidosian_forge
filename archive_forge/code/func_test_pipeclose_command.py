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
def test_pipeclose_command(self):
    app, command = make_app()
    result = app.run(['pipe-close'])
    self.assertEqual(result, 141)