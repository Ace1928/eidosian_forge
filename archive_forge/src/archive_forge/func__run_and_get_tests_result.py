from collections import abc
import contextlib
import dataclasses
import difflib
import enum
import errno
import faulthandler
import getpass
import inspect
import io
import itertools
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import typing
from typing import Any, AnyStr, BinaryIO, Callable, ContextManager, IO, Iterator, List, Mapping, MutableMapping, MutableSequence, NoReturn, Optional, Sequence, Text, TextIO, Tuple, Type, Union
import unittest
from unittest import mock  # pylint: disable=unused-import Allow absltest.mock.
from urllib import parse
from absl import app  # pylint: disable=g-import-not-at-top
from absl import flags
from absl import logging
from absl.testing import _pretty_print_reporter
from absl.testing import xml_reporter
def _run_and_get_tests_result(argv: MutableSequence[str], args: Sequence[Any], kwargs: MutableMapping[str, Any], xml_test_runner_class: Type[unittest.TextTestRunner]) -> Tuple[unittest.TestResult, bool]:
    """Same as run_tests, but it doesn't exit.

  Args:
    argv: sys.argv with the command-line flags removed from the front, i.e. the
      argv with which :func:`app.run()<absl.app.run>` has called
      ``__main__.main``. It is passed to
      ``unittest.TestProgram.__init__(argv=)``, which does its own flag parsing.
      It is ignored if kwargs contains an argv entry.
    args: Positional arguments passed through to
      ``unittest.TestProgram.__init__``.
    kwargs: Keyword arguments passed through to
      ``unittest.TestProgram.__init__``.
    xml_test_runner_class: The type of the test runner class.

  Returns:
    A tuple of ``(test_result, fail_when_no_tests_ran)``.
    ``fail_when_no_tests_ran`` indicates whether the test should fail when
    no tests ran.
  """
    argv = kwargs.pop('argv', argv)
    if sys.version_info[:2] >= (3, 12):
        fail_when_no_tests_ran = True
    else:
        fail_when_no_tests_ran = False
    if _setup_filtering(argv):
        fail_when_no_tests_ran = False
    _setup_test_runner_fail_fast(argv)
    kwargs['testLoader'], shard_index = _setup_sharding(kwargs.get('testLoader', None))
    if shard_index is not None and shard_index > 0:
        fail_when_no_tests_ran = False
    if not FLAGS.xml_output_file:
        FLAGS.xml_output_file = get_default_xml_output_filename()
    xml_output_file = FLAGS.xml_output_file
    xml_buffer = None
    if xml_output_file:
        xml_output_dir = os.path.dirname(xml_output_file)
        if xml_output_dir and (not os.path.isdir(xml_output_dir)):
            try:
                os.makedirs(xml_output_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with _open(xml_output_file, 'w'):
            pass
        if kwargs.get('testRunner') is not None and (not hasattr(kwargs['testRunner'], 'set_default_xml_stream')):
            sys.stderr.write('WARNING: XML_OUTPUT_FILE or --xml_output_file setting overrides testRunner=%r setting (possibly from --pdb)' % kwargs['testRunner'])
            kwargs['testRunner'] = xml_test_runner_class
        if kwargs.get('testRunner') is None:
            kwargs['testRunner'] = xml_test_runner_class
        xml_buffer = io.StringIO()
        kwargs['testRunner'].set_default_xml_stream(xml_buffer)
        randomize_ordering_seed = getattr(kwargs['testLoader'], '_randomize_ordering_seed', None)
        setter = getattr(kwargs['testRunner'], 'set_testsuites_property', None)
        if randomize_ordering_seed and setter:
            setter('test_randomize_ordering_seed', randomize_ordering_seed)
    elif kwargs.get('testRunner') is None:
        kwargs['testRunner'] = _pretty_print_reporter.TextTestRunner
    if FLAGS.pdb_post_mortem:
        runner = kwargs['testRunner']
        if isinstance(runner, type) and issubclass(runner, _pretty_print_reporter.TextTestRunner) or isinstance(runner, _pretty_print_reporter.TextTestRunner):
            runner.run_for_debugging = True
    if not os.path.isdir(TEST_TMPDIR.value):
        try:
            os.makedirs(TEST_TMPDIR.value)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    kwargs['argv'] = argv
    kwargs['exit'] = False
    try:
        test_program = unittest.TestProgram(*args, **kwargs)
        return (test_program.result, fail_when_no_tests_ran)
    finally:
        if xml_buffer:
            try:
                with _open(xml_output_file, 'w') as f:
                    f.write(xml_buffer.getvalue())
            finally:
                xml_buffer.close()