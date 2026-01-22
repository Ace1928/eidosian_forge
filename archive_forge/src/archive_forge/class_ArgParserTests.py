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
class ArgParserTests(TestCase):

    def test_can_parse_attach_file_without_test_id(self):
        with NamedTemporaryFile() as tmp_file:
            args = safe_parse_arguments(args=['--attach-file', tmp_file.name])
            self.assertThat(args.attach_file.name, Equals(tmp_file.name))

    def test_can_run_without_args(self):
        safe_parse_arguments([])

    def test_cannot_specify_more_than_one_status_command(self):

        def fn():
            return safe_parse_arguments(['--fail', 'foo', '--skip', 'bar'])
        self.assertThat(fn, raises(RuntimeError('argument --skip: Only one status may be specified at once.')))

    def test_cannot_specify_mimetype_without_attach_file(self):

        def fn():
            return safe_parse_arguments(['--mimetype', 'foo'])
        self.assertThat(fn, raises(RuntimeError('Cannot specify --mimetype without --attach-file')))

    def test_cannot_specify_filename_without_attach_file(self):

        def fn():
            return safe_parse_arguments(['--file-name', 'foo'])
        self.assertThat(fn, raises(RuntimeError('Cannot specify --file-name without --attach-file')))

    def test_can_specify_tags_without_status_command(self):
        args = safe_parse_arguments(['--tag', 'foo'])
        self.assertEqual(['foo'], args.tags)

    def test_must_specify_tags_with_tags_options(self):

        def fn():
            return safe_parse_arguments(['--fail', 'foo', '--tag'])
        self.assertThat(fn, MatchesAny(raises(RuntimeError('--tag option requires 1 argument')), raises(RuntimeError('--tag option requires an argument'))))