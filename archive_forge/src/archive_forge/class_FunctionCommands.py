import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class FunctionCommands:
    """
    Redis Function commands
    """

    def function_load(self, code: str, replace: Optional[bool]=False) -> Union[Awaitable[str], str]:
        """
        Load a library to Redis.
        :param code: the source code (must start with
        Shebang statement that provides a metadata about the library)
        :param replace: changes the behavior to overwrite the existing library
        with the new contents.
        Return the library name that was loaded.

        For more information see https://redis.io/commands/function-load
        """
        pieces = ['REPLACE'] if replace else []
        pieces.append(code)
        return self.execute_command('FUNCTION LOAD', *pieces)

    def function_delete(self, library: str) -> Union[Awaitable[str], str]:
        """
        Delete the library called ``library`` and all its functions.

        For more information see https://redis.io/commands/function-delete
        """
        return self.execute_command('FUNCTION DELETE', library)

    def function_flush(self, mode: str='SYNC') -> Union[Awaitable[str], str]:
        """
        Deletes all the libraries.

        For more information see https://redis.io/commands/function-flush
        """
        return self.execute_command('FUNCTION FLUSH', mode)

    def function_list(self, library: Optional[str]='*', withcode: Optional[bool]=False) -> Union[Awaitable[List], List]:
        """
        Return information about the functions and libraries.
        :param library: pecify a pattern for matching library names
        :param withcode: cause the server to include the libraries source
         implementation in the reply
        """
        args = ['LIBRARYNAME', library]
        if withcode:
            args.append('WITHCODE')
        return self.execute_command('FUNCTION LIST', *args)

    def _fcall(self, command: str, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
        return self.execute_command(command, function, numkeys, *keys_and_args)

    def fcall(self, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
        """
        Invoke a function.

        For more information see https://redis.io/commands/fcall
        """
        return self._fcall('FCALL', function, numkeys, *keys_and_args)

    def fcall_ro(self, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
        """
        This is a read-only variant of the FCALL command that cannot
        execute commands that modify data.

        For more information see https://redis.io/commands/fcal_ro
        """
        return self._fcall('FCALL_RO', function, numkeys, *keys_and_args)

    def function_dump(self) -> Union[Awaitable[str], str]:
        """
        Return the serialized payload of loaded libraries.

        For more information see https://redis.io/commands/function-dump
        """
        from redis.client import NEVER_DECODE
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command('FUNCTION DUMP', **options)

    def function_restore(self, payload: str, policy: Optional[str]='APPEND') -> Union[Awaitable[str], str]:
        """
        Restore libraries from the serialized ``payload``.
        You can use the optional policy argument to provide a policy
        for handling existing libraries.

        For more information see https://redis.io/commands/function-restore
        """
        return self.execute_command('FUNCTION RESTORE', payload, policy)

    def function_kill(self) -> Union[Awaitable[str], str]:
        """
        Kill a function that is currently executing.

        For more information see https://redis.io/commands/function-kill
        """
        return self.execute_command('FUNCTION KILL')

    def function_stats(self) -> Union[Awaitable[List], List]:
        """
        Return information about the function that's currently running
        and information about the available execution engines.

        For more information see https://redis.io/commands/function-stats
        """
        return self.execute_command('FUNCTION STATS')