from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class OperationFailure(PyMongoError):
    """Raised when a database operation fails.

    .. versionadded:: 2.7
       The :attr:`details` attribute.
    """

    def __init__(self, error: str, code: Optional[int]=None, details: Optional[Mapping[str, Any]]=None, max_wire_version: Optional[int]=None) -> None:
        error_labels = None
        if details is not None:
            error_labels = details.get('errorLabels')
        super().__init__(_format_detailed_error(error, details), error_labels=error_labels)
        self.__code = code
        self.__details = details
        self.__max_wire_version = max_wire_version

    @property
    def _max_wire_version(self) -> Optional[int]:
        return self.__max_wire_version

    @property
    def code(self) -> Optional[int]:
        """The error code returned by the server, if any."""
        return self.__code

    @property
    def details(self) -> Optional[Mapping[str, Any]]:
        """The complete error document returned by the server.

        Depending on the error that occurred, the error document
        may include useful information beyond just the error
        message. When connected to a mongos the error document
        may contain one or more subdocuments if errors occurred
        on multiple shards.
        """
        return self.__details

    @property
    def timeout(self) -> bool:
        return self.__code in (50,)