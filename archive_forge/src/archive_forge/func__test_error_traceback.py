from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def _test_error_traceback(case, traceback_matcher):
    """Match result log of single test that errored out.

    ``traceback_matcher`` is applied to the text of the traceback.
    """
    return MatchesListwise([Equals(('startTest', case)), MatchesListwise([Equals('addError'), Equals(case), MatchesDict({'traceback': AfterPreprocessing(lambda x: x.as_text(), traceback_matcher)})]), Equals(('stopTest', case))])