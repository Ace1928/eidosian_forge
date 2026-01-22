import re
import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator, StreamToExtendedDecorator
from subunit import StreamResultToBytes, read_test_list
from subunit.filters import filter_by_result, find_stream
from subunit.test_results import (TestResultFilter, and_predicates,
def _make_result(output, options, predicate):
    """Make the result that we'll send the test outcomes to."""
    fixup_expected_failures = set()
    for path in options.fixup_expected_failures or ():
        fixup_expected_failures.update(read_test_list(path))
    return StreamToExtendedDecorator(TestResultFilter(ExtendedToStreamDecorator(StreamResultToBytes(output)), filter_error=options.error, filter_failure=options.failure, filter_success=options.success, filter_skip=options.skip, filter_xfail=options.xfail, filter_predicate=predicate, fixup_expected_failures=fixup_expected_failures, rename=_compile_rename(options.renames)))