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
def assertContainsExactSubsequence(self, container, subsequence, msg=None):
    """Asserts that "container" contains "subsequence" as an exact subsequence.

    Asserts that "container" contains all the elements of "subsequence", in
    order, and without other elements interspersed. For example, [1, 2, 3] is an
    exact subsequence of [0, 0, 1, 2, 3, 0] but not of [0, 0, 1, 2, 0, 3, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be an exact subsequence of container.
      msg: Optional message to report on failure.
    """
    container = list(container)
    subsequence = list(subsequence)
    longest_match = 0
    for start in range(1 + len(container) - len(subsequence)):
        if longest_match == len(subsequence):
            break
        index = 0
        while index < len(subsequence) and subsequence[index] == container[start + index]:
            index += 1
        longest_match = max(longest_match, index)
    if longest_match < len(subsequence):
        self.fail('%s not an exact subsequence of %s. Longest matching prefix: %s' % (subsequence, container, subsequence[:longest_match]), msg)