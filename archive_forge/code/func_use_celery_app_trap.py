import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def use_celery_app_trap():
    """You can override this fixture to enable the app trap.

    The app trap raises an exception whenever something attempts
    to use the current or default apps.
    """
    return False