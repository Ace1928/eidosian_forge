from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class CommandStartedEvent(_CommandEvent):
    """Event published when a command starts.

    :Parameters:
      - `command`: The command document.
      - `database_name`: The name of the database this command was run against.
      - `request_id`: The request id for this operation.
      - `connection_id`: The address (host, port) of the server this command
        was sent to.
      - `operation_id`: An optional identifier for a series of related events.
      - `service_id`: The service_id this command was sent to, or ``None``.
    """
    __slots__ = ('__cmd',)

    def __init__(self, command: _DocumentOut, database_name: str, request_id: int, connection_id: _Address, operation_id: Optional[int], service_id: Optional[ObjectId]=None) -> None:
        if not command:
            raise ValueError(f'{command!r} is not a valid command')
        command_name = next(iter(command))
        super().__init__(command_name, request_id, connection_id, operation_id, service_id=service_id, database_name=database_name)
        cmd_name = command_name.lower()
        if cmd_name in _SENSITIVE_COMMANDS or _is_speculative_authenticate(cmd_name, command):
            self.__cmd: _DocumentOut = {}
        else:
            self.__cmd = command

    @property
    def command(self) -> _DocumentOut:
        """The command document."""
        return self.__cmd

    @property
    def database_name(self) -> str:
        """The name of the database this command was run against."""
        return super().database_name

    def __repr__(self) -> str:
        return '<{} {} db: {!r}, command: {!r}, operation_id: {}, service_id: {}>'.format(self.__class__.__name__, self.connection_id, self.database_name, self.command_name, self.operation_id, self.service_id)