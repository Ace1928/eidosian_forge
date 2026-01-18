import os
import sys
import tempfile
from os import environ as env
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import pytest
from .. import data as nibd
from ..data import (
from .test_environment import DATA_KEY, USER_KEY, with_environment
@pytest.fixture
def with_nimd_env(request, with_environment):
    DATA_FUNCS = {}
    DATA_FUNCS['home_dir_func'] = nibd.get_nipy_user_dir
    DATA_FUNCS['sys_dir_func'] = nibd.get_nipy_system_dir
    DATA_FUNCS['path_func'] = nibd.get_data_path
    yield
    nibd.get_nipy_user_dir = DATA_FUNCS['home_dir_func']
    nibd.get_nipy_system_dir = DATA_FUNCS['sys_dir_func']
    nibd.get_data_path = DATA_FUNCS['path_func']