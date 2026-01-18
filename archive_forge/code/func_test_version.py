import errno
import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import unittest
from textwrap import dedent
from bpython import args
from bpython.config import getpreferredencoding
from bpython.test import FixLanguageTestCase as TestCase
def test_version(self):
    with self.assertRaises(SystemExit):
        args.parse(['--version'])