import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_report_activity(self):
    activity = []

    def log_activity(length, direction):
        activity.append((length, direction))
    from_file = BytesIO(self.test_data)
    to_file = BytesIO()
    osutils.pumpfile(from_file, to_file, buff_size=500, report_activity=log_activity, direction='read')
    self.assertEqual([(500, 'read'), (500, 'read'), (500, 'read'), (36, 'read')], activity)
    from_file = BytesIO(self.test_data)
    to_file = BytesIO()
    del activity[:]
    osutils.pumpfile(from_file, to_file, buff_size=500, report_activity=log_activity, direction='write')
    self.assertEqual([(500, 'write'), (500, 'write'), (500, 'write'), (36, 'write')], activity)
    from_file = BytesIO(self.test_data)
    to_file = BytesIO()
    del activity[:]
    osutils.pumpfile(from_file, to_file, buff_size=500, read_length=1028, report_activity=log_activity, direction='read')
    self.assertEqual([(500, 'read'), (500, 'read'), (28, 'read')], activity)