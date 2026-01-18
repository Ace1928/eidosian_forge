import os
from os import environ as env
from os.path import abspath
from os.path import join as pjoin
import pytest
from .. import environment as nibe
@pytest.fixture
def with_environment(request):
    """Setup test environment for some functions that are tested
    in this module. In particular this functions stores attributes
    and other things that we need to stub in some test functions.
    This needs to be done on a function level and not module level because
    each testfunction needs a pristine environment.
    """
    GIVEN_ENV = {}
    GIVEN_ENV['env'] = env.copy()
    yield
    'Restore things that were remembered by the setup_environment function '
    orig_env = GIVEN_ENV['env']
    for key in list(env.keys()):
        if key not in orig_env:
            del env[key]
    env.update(orig_env)