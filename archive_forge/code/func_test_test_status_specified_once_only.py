import datetime
import optparse
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
from tempfile import NamedTemporaryFile
from iso8601 import UTC
from testtools import TestCase
from testtools.matchers import (Equals, Matcher, MatchesAny, MatchesListwise,
from testtools.testresult.doubles import StreamResult
import subunit._output as _o
from subunit._output import (_ALL_ACTIONS, _FINAL_ACTIONS,
def test_test_status_specified_once_only(self):
    with temp_file_contents(b'Hi') as f:
        self.patch(_o, '_CHUNK_SIZE', 1)
        result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
        if self.status in _FINAL_ACTIONS:
            first_call = MatchesStatusCall(test_id=self.test_id, test_status=None)
            last_call = MatchesStatusCall(test_id=self.test_id, test_status=self.status)
        else:
            first_call = MatchesStatusCall(test_id=self.test_id, test_status=self.status)
            last_call = MatchesStatusCall(test_id=self.test_id, test_status=None)
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), first_call, last_call, MatchesStatusCall(call='stopTestRun')]))