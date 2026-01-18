import re
import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator, StreamToExtendedDecorator
from subunit import StreamResultToBytes, read_test_list
from subunit.filters import filter_by_result, find_stream
from subunit.test_results import (TestResultFilter, and_predicates,
Check if this test and error match the regexp filters.