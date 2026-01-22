from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union
class PinnedResponse(Response):
    __slots__ = ('_conn', '_more_to_come')

    def __init__(self, data: Union[_OpMsg, _OpReply], address: _Address, conn: Connection, request_id: int, duration: Optional[timedelta], from_command: bool, docs: list[_DocumentOut], more_to_come: bool):
        """Represent a response to an exhaust cursor's initial query.

        :Parameters:
          - `data`:  A network response message.
          - `address`: (host, port) of the source server.
          - `conn`: The Connection used for the initial query.
          - `request_id`: The request id of this operation.
          - `duration`: The duration of the operation.
          - `from_command`: If the response is the result of a db command.
          - `docs`: List of documents.
          - `more_to_come`: Bool indicating whether cursor is ready to be
            exhausted.
        """
        super().__init__(data, address, request_id, duration, from_command, docs)
        self._conn = conn
        self._more_to_come = more_to_come

    @property
    def conn(self) -> Connection:
        """The Connection used for the initial query.

        The server will send batches on this socket, without waiting for
        getMores from the client, until the result set is exhausted or there
        is an error.
        """
        return self._conn

    @property
    def more_to_come(self) -> bool:
        """If true, server is ready to send batches on the socket until the
        result set is exhausted or there is an error.
        """
        return self._more_to_come