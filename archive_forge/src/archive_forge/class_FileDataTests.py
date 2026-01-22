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
class FileDataTests(TestCase):

    def test_can_attach_file_without_test_id(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for(['--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=None, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_file_name_is_used_by_default(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for(['--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=f.name, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_filename_can_be_overridden(self):
        with temp_file_contents(b'Hello') as f:
            specified_file_name = self.getUniqueString()
            result = get_result_for(['--attach-file', f.name, '--file-name', specified_file_name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=specified_file_name, file_bytes=b'Hello'), MatchesStatusCall(call='stopTestRun')]))

    def test_files_have_timestamp(self):
        _dummy_timestamp = datetime.datetime(2013, 1, 1, 0, 0, 0, 0, UTC)
        self.patch(_o, 'create_timestamp', lambda: _dummy_timestamp)
        with temp_file_contents(b'Hello') as f:
            self.getUniqueString()
            result = get_result_for(['--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'Hello', timestamp=_dummy_timestamp), MatchesStatusCall(call='stopTestRun')]))

    def test_can_specify_tags_without_test_status(self):
        result = get_result_for(['--tag', 'foo'])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_tags={'foo'}), MatchesStatusCall(call='stopTestRun')]))