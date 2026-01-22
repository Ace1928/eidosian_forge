import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class AsyncManagementCommands(ManagementCommands):

    async def command_info(self, **kwargs) -> None:
        return super().command_info(**kwargs)

    async def debug_segfault(self, **kwargs) -> None:
        return super().debug_segfault(**kwargs)

    async def memory_doctor(self, **kwargs) -> None:
        return super().memory_doctor(**kwargs)

    async def memory_help(self, **kwargs) -> None:
        return super().memory_help(**kwargs)

    async def shutdown(self, save: bool=False, nosave: bool=False, now: bool=False, force: bool=False, abort: bool=False, **kwargs) -> None:
        """Shutdown the Redis server.  If Redis has persistence configured,
        data will be flushed before shutdown.  If the "save" option is set,
        a data flush will be attempted even if there is no persistence
        configured.  If the "nosave" option is set, no data flush will be
        attempted.  The "save" and "nosave" options cannot both be set.

        For more information see https://redis.io/commands/shutdown
        """
        if save and nosave:
            raise DataError('SHUTDOWN save and nosave cannot both be set')
        args = ['SHUTDOWN']
        if save:
            args.append('SAVE')
        if nosave:
            args.append('NOSAVE')
        if now:
            args.append('NOW')
        if force:
            args.append('FORCE')
        if abort:
            args.append('ABORT')
        try:
            await self.execute_command(*args, **kwargs)
        except ConnectionError:
            return
        raise RedisError('SHUTDOWN seems to have failed.')