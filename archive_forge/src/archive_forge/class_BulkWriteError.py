from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class BulkWriteError(OperationFailure):
    """Exception class for bulk write errors.

    .. versionadded:: 2.7
    """
    details: _DocumentOut

    def __init__(self, results: _DocumentOut) -> None:
        super().__init__('batch op errors occurred', 65, results)

    def __reduce__(self) -> tuple[Any, Any]:
        return (self.__class__, (self.details,))

    @property
    def timeout(self) -> bool:
        wces = self.details.get('writeConcernErrors', [])
        if wces and _wtimeout_error(wces[-1]):
            return True
        werrs = self.details.get('writeErrors', [])
        if werrs and werrs[-1].get('code') == 50:
            return True
        return False