import sys
import unittest
from operator import methodcaller
from optparse import OptionParser
from testtools import (DecorateTestCaseResult, StreamResultRouter,
from subunit import ByteStreamToStreamResult
from subunit.filters import find_stream
from subunit.test_results import CatFiles
Display a subunit stream through python's unittest test runner.