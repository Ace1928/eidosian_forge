from typing import Any
from triad import assert_or_throw
from triad.utils.hash import to_uuid
class Yielded(object):
    """Yields from :class:`~fugue.workflow.workflow.FugueWorkflow`.
    Users shouldn't create this object directly.

    :param yid: unique id for determinism
    """

    def __init__(self, yid: str):
        self._yid = to_uuid(yid)

    def __uuid__(self) -> str:
        """uuid of the instance"""
        return self._yid

    @property
    def is_set(self) -> bool:
        """Whether the value is set. It can be false if the parent workflow
        has not been executed.
        """
        raise NotImplementedError

    def __copy__(self) -> Any:
        """``copy`` should have no effect"""
        return self

    def __deepcopy__(self, memo: Any) -> Any:
        """``deepcopy`` should have no effect"""
        return self