import os
from os import environ as env
from os.path import abspath
from os.path import join as pjoin
import pytest
from .. import environment as nibe
def test_user_dir(with_environment):
    if USER_KEY in env:
        del env[USER_KEY]
    home_dir = nibe.get_home_dir()
    if os.name == 'posix':
        exp = pjoin(home_dir, '.nipy')
    else:
        exp = pjoin(home_dir, '_nipy')
    assert exp == nibe.get_nipy_user_dir()
    env[USER_KEY] = '/a/path'
    assert abspath('/a/path') == nibe.get_nipy_user_dir()