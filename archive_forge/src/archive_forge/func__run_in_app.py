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
def _run_in_app(function, args, kwargs):
    """Executes a set of Python unit tests, ensuring app.run.

  This is a private function, users should call absltest.main().

  _run_in_app calculates argv to be the command-line arguments of this program
  (without the flags), sets the default of FLAGS.alsologtostderr to True,
  then it calls function(argv, args, kwargs), making sure that `function'
  will get called within app.run(). _run_in_app does this by checking whether
  it is called by app.run(), or by calling app.run() explicitly.

  The reason why app.run has to be ensured is to make sure that
  flags are parsed and stripped properly, and other initializations done by
  the app module are also carried out, no matter if absltest.run() is called
  from within or outside app.run().

  If _run_in_app is called from within app.run(), then it will reparse
  sys.argv and pass the result without command-line flags into the argv
  argument of `function'. The reason why this parsing is needed is that
  __main__.main() calls absltest.main() without passing its argv. So the
  only way _run_in_app could get to know the argv without the flags is that
  it reparses sys.argv.

  _run_in_app changes the default of FLAGS.alsologtostderr to True so that the
  test program's stderr will contain all the log messages unless otherwise
  specified on the command-line. This overrides any explicit assignment to
  FLAGS.alsologtostderr by the test program prior to the call to _run_in_app()
  (e.g. in __main__.main).

  Please note that _run_in_app (and the function it calls) is allowed to make
  changes to kwargs.

  Args:
    function: absltest.run_tests or a similar function. It will be called as
        function(argv, args, kwargs) where argv is a list containing the
        elements of sys.argv without the command-line flags.
    args: Positional arguments passed through to unittest.TestProgram.__init__.
    kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
    if _is_in_app_main():
        _register_sigterm_with_faulthandler()
        FLAGS.set_default('alsologtostderr', True)
        stored_parse_methods = {}
        noop_parse = lambda _: None
        for name in FLAGS:
            stored_parse_methods[name] = FLAGS[name].parse
        for name in FLAGS:
            FLAGS[name].parse = noop_parse
        try:
            argv = FLAGS(sys.argv)
        finally:
            for name in FLAGS:
                FLAGS[name].parse = stored_parse_methods[name]
            sys.stdout.flush()
        function(argv, args, kwargs)
    else:
        FLAGS.set_default('alsologtostderr', True)

        def main_function(argv):
            _register_sigterm_with_faulthandler()
            function(argv, args, kwargs)
        app.run(main=main_function)