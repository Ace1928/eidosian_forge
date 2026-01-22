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
def _maybe_add_temp_path_cleanup(self, path, cleanup):
    cleanup = self._get_tempfile_cleanup(cleanup)
    if cleanup == TempFileCleanup.OFF:
        return
    elif cleanup == TempFileCleanup.ALWAYS:
        self.addCleanup(_rmtree_ignore_errors, path)
    elif cleanup == TempFileCleanup.SUCCESS:
        self._internal_add_cleanup_on_success(_rmtree_ignore_errors, path)
    else:
        raise AssertionError('Unexpected cleanup value: {}'.format(cleanup))