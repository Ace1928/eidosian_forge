from . import events
from . import exceptions
from . import tasks
def _is_base_error(self, exc: BaseException) -> bool:
    assert isinstance(exc, BaseException)
    return isinstance(exc, (SystemExit, KeyboardInterrupt))