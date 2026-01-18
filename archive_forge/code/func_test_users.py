from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
@docker_available
def test_users():
    x = Interface(config='.xnat.cfg')
    assert isinstance(x.manage.users(), list)