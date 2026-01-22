import sys
import traceback
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import _pytest._code
from _pytest.compat import getimfunc
from _pytest.compat import is_async_function
from _pytest.config import hookimpl
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Module
from _pytest.runner import CallInfo
import pytest
def _register_unittest_setup_class_fixture(self, cls: type) -> None:
    """Register an auto-use fixture to invoke setUpClass and
        tearDownClass (#517)."""
    setup = getattr(cls, 'setUpClass', None)
    teardown = getattr(cls, 'tearDownClass', None)
    if setup is None and teardown is None:
        return None
    cleanup = getattr(cls, 'doClassCleanups', lambda: None)

    def unittest_setup_class_fixture(request: FixtureRequest) -> Generator[None, None, None]:
        cls = request.cls
        if _is_skipped(cls):
            reason = cls.__unittest_skip_why__
            raise pytest.skip.Exception(reason, _use_item_location=True)
        if setup is not None:
            try:
                setup()
            except Exception:
                cleanup()
                raise
        yield
        try:
            if teardown is not None:
                teardown()
        finally:
            cleanup()
    self.session._fixturemanager._register_fixture(name=f'_unittest_setUpClass_fixture_{cls.__qualname__}', func=unittest_setup_class_fixture, nodeid=self.nodeid, scope='class', autouse=True)