import re
import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator, StreamToExtendedDecorator
from subunit import StreamResultToBytes, read_test_list
from subunit.filters import filter_by_result, find_stream
from subunit.test_results import (TestResultFilter, and_predicates,
def _compile_rename(patterns):

    def rename(name):
        for from_pattern, to_pattern in patterns:
            name = re.sub(from_pattern, to_pattern, name)
        return name
    return rename