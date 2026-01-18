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
def test_saveKeyNoFilename(self) -> None:
    """
        When no path is specified, it will ask for the path used to store the
        key.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    keyPath = base.child('custom_key').path
    input_prompts: list[str] = []
    import twisted.conch.scripts.ckeygen

    def mock_input(*args: object) -> str:
        input_prompts.append('')
        return ''
    self.patch(twisted.conch.scripts.ckeygen, '_inputSaveFile', lambda _: keyPath)
    key = Key.fromString(privateRSA_openssh)
    _saveKey(key, {'filename': None, 'no-passphrase': True, 'format': 'md5-hex'}, mock_input)
    persistedKeyContent = base.child('custom_key').getContent()
    persistedKey = key.fromString(persistedKeyContent, None, b'')
    self.assertEqual(key, persistedKey)