import io
import json
import optparse
import os.path
import sys
from errno import EEXIST
from textwrap import dedent
from testtools import StreamToDict
from subunit.filters import run_tests_from_stream
Exports tests to disk.