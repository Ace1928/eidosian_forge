import os
from shutil import rmtree
import pytest
from nipype.utils.misc import (
@pytest.mark.parametrize('string, expected', [('yes', True), ('true', True), ('t', True), ('1', True), ('no', False), ('false', False), ('n', False), ('f', False), ('0', False)])
def test_str2bool(string, expected):
    assert str2bool(string) == expected