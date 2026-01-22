from contextlib import contextmanager
from contextlib import nullcontext
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import io
from io import StringIO
import logging
from logging import LogRecord
import os
from pathlib import Path
import re
from types import TracebackType
from typing import AbstractSet
from typing import Dict
from typing import final
from typing import Generator
from typing import Generic
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.capture import CaptureManager
from _pytest.config import _strtobool
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
class _LiveLoggingStreamHandler(logging_StreamHandler):
    """A logging StreamHandler used by the live logging feature: it will
    write a newline before the first log message in each test.

    During live logging we must also explicitly disable stdout/stderr
    capturing otherwise it will get captured and won't appear in the
    terminal.
    """
    stream: TerminalReporter = None

    def __init__(self, terminal_reporter: TerminalReporter, capture_manager: Optional[CaptureManager]) -> None:
        super().__init__(stream=terminal_reporter)
        self.capture_manager = capture_manager
        self.reset()
        self.set_when(None)
        self._test_outcome_written = False

    def reset(self) -> None:
        """Reset the handler; should be called before the start of each test."""
        self._first_record_emitted = False

    def set_when(self, when: Optional[str]) -> None:
        """Prepare for the given test phase (setup/call/teardown)."""
        self._when = when
        self._section_name_shown = False
        if when == 'start':
            self._test_outcome_written = False

    def emit(self, record: logging.LogRecord) -> None:
        ctx_manager = self.capture_manager.global_and_fixture_disabled() if self.capture_manager else nullcontext()
        with ctx_manager:
            if not self._first_record_emitted:
                self.stream.write('\n')
                self._first_record_emitted = True
            elif self._when in ('teardown', 'finish'):
                if not self._test_outcome_written:
                    self._test_outcome_written = True
                    self.stream.write('\n')
            if not self._section_name_shown and self._when:
                self.stream.section('live log ' + self._when, sep='-', bold=True)
                self._section_name_shown = True
            super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        pass