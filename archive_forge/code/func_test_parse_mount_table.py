import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('output, exit_code, expected', MOUNT_OUTPUTS)
def test_parse_mount_table(output, exit_code, expected):
    assert _parse_mount_table(exit_code, output) == expected