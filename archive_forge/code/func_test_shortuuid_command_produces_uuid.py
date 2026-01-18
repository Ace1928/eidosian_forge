import os
import string
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4
from shortuuid.cli import cli
from shortuuid.main import decode
from shortuuid.main import encode
from shortuuid.main import get_alphabet
from shortuuid.main import random
from shortuuid.main import set_alphabet
from shortuuid.main import ShortUUID
from shortuuid.main import uuid
@patch('shortuuid.cli.print')
def test_shortuuid_command_produces_uuid(self, mock_print):
    cli([])
    mock_print.assert_called()
    terminal_output = mock_print.call_args[0][0]
    self.assertEqual(len(terminal_output), 22)