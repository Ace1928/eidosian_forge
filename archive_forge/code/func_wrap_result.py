import sys
import unittest
from operator import methodcaller
from optparse import OptionParser
from testtools import (DecorateTestCaseResult, StreamResultRouter,
from subunit import ByteStreamToStreamResult
from subunit.filters import find_stream
from subunit.test_results import CatFiles
def wrap_result(result):
    result = StreamToExtendedDecorator(result)
    if not options.no_passthrough:
        result = StreamResultRouter(result)
        result.add_rule(CatFiles(sys.stdout), 'test_id', test_id=None)
    return result