from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import (
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_useDefaultForKey(self) -> None:
    """
        L{options} will default to "~/.ssh/id_rsa" if the user doesn't
        specify a key.
        """
    input_prompts: list[str] = []

    def mock_input(*args: object) -> str:
        input_prompts.append('')
        return ''
    options = {'filename': ''}
    filename = _getKeyOrDefault(options, mock_input)
    self.assertEqual(options['filename'], '')
    self.assertTrue(filename.endswith(os.path.join('.ssh', 'id_rsa')))
    self.assertEqual(1, len(input_prompts))
    self.assertEqual([''], input_prompts)