import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
class PagedStreamTest(fixtures.TestWithFixtures):

    def setUp(self) -> None:
        out = sinks.TTYFixture()
        self.useFixture(out)
        self.stream = out.stream
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', self.stream))
        popen = fixtures.MockPatch('subprocess.Popen')
        self.useFixture(popen)
        self.popen = popen.mock

    def test_defaults(self) -> None:

        class TestCommand(command.PagerCommand):

            def command(self) -> List[str]:
                return []

            def environment_variables(self, config: _PagerConfig) -> Optional[Dict[str, str]]:
                return None
        tc = TestCommand()
        ap = autopage.AutoPager(pager_command=tc, line_buffering=False)
        with mock.patch.object(ap, '_pager_env') as get_env, mock.patch.object(tc, 'command') as cmd:
            stream = ap._paged_stream()
            self.popen.assert_called_once_with(cmd.return_value, env=get_env.return_value, bufsize=-1, universal_newlines=True, encoding='UTF-8', errors='strict', stdin=subprocess.PIPE, stdout=None)
            self.assertIs(stream, self.popen.return_value.stdin)

    def test_defaults_cmd_as_class(self) -> None:

        class TestCommand(command.PagerCommand):

            def command(self) -> List[str]:
                return []

            def environment_variables(self, config: _PagerConfig) -> Optional[Dict[str, str]]:
                return None
        with mock.patch.object(TestCommand, 'command') as cmd:
            ap = autopage.AutoPager(pager_command=TestCommand, line_buffering=False)
            with mock.patch.object(ap, '_pager_env') as get_env:
                stream = ap._paged_stream()
                self.popen.assert_called_once_with(cmd.return_value, env=get_env.return_value, bufsize=-1, universal_newlines=True, encoding='UTF-8', errors='strict', stdin=subprocess.PIPE, stdout=None)
                self.assertIs(stream, self.popen.return_value.stdin)

    def test_defaults_cmd_as_string(self) -> None:
        ap = autopage.AutoPager(pager_command='foo bar', line_buffering=False)
        with mock.patch.object(ap, '_pager_env') as get_env:
            stream = ap._paged_stream()
            self.popen.assert_called_once_with(['foo', 'bar'], env=get_env.return_value, bufsize=-1, universal_newlines=True, encoding='UTF-8', errors='strict', stdin=subprocess.PIPE, stdout=None)
            self.assertIs(stream, self.popen.return_value.stdin)

    def test_defaults_cmd_as_int(self) -> None:
        self.assertRaises(TypeError, autopage.AutoPager, pager_command=42)

    def test_line_buffering(self) -> None:
        ap = autopage.AutoPager(line_buffering=True)
        stream = ap._paged_stream()
        self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=1, universal_newlines=True, encoding=mock.ANY, errors=mock.ANY, stdin=subprocess.PIPE, stdout=None)
        self.assertIs(stream, self.popen.return_value.stdin)

    def test_errors(self) -> None:
        ap = autopage.AutoPager(errors=autopage.ErrorStrategy.NAME_REPLACE)
        stream = ap._paged_stream()
        self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=mock.ANY, universal_newlines=mock.ANY, encoding=mock.ANY, errors='namereplace', stdin=subprocess.PIPE, stdout=None)
        self.assertIs(stream, self.popen.return_value.stdin)

    def test_explicit_stdout_stream(self) -> None:
        ap = autopage.AutoPager(self.stream)
        stream = ap._paged_stream()
        self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=mock.ANY, universal_newlines=mock.ANY, encoding=mock.ANY, errors=mock.ANY, stdin=subprocess.PIPE, stdout=None)
        self.assertIs(stream, self.popen.return_value.stdin)

    def test_explicit_stream(self) -> None:
        with sinks.TTYFixture() as tty:
            ap = autopage.AutoPager(tty.stream)
            stream = ap._paged_stream()
            self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=mock.ANY, universal_newlines=mock.ANY, encoding=mock.ANY, errors=mock.ANY, stdin=subprocess.PIPE, stdout=tty.stream)
            self.assertIs(stream, self.popen.return_value.stdin)