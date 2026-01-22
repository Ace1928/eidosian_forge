import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class GearsCommands:

    def tfunction_load(self, lib_code: str, replace: bool=False, config: Union[str, None]=None) -> ResponseT:
        """
        Load a new library to RedisGears.

        ``lib_code`` - the library code.
        ``config`` - a string representation of a JSON object
        that will be provided to the library on load time,
        for more information refer to
        https://github.com/RedisGears/RedisGears/blob/master/docs/function_advance_topics.md#library-configuration
        ``replace`` - an optional argument, instructs RedisGears to replace the
        function if its already exists

        For more information see https://redis.io/commands/tfunction-load/
        """
        pieces = []
        if replace:
            pieces.append('REPLACE')
        if config is not None:
            pieces.extend(['CONFIG', config])
        pieces.append(lib_code)
        return self.execute_command('TFUNCTION LOAD', *pieces)

    def tfunction_delete(self, lib_name: str) -> ResponseT:
        """
        Delete a library from RedisGears.

        ``lib_name`` the library name to delete.

        For more information see https://redis.io/commands/tfunction-delete/
        """
        return self.execute_command('TFUNCTION DELETE', lib_name)

    def tfunction_list(self, with_code: bool=False, verbose: int=0, lib_name: Union[str, None]=None) -> ResponseT:
        """
        List the functions with additional information about each function.

        ``with_code`` Show libraries code.
        ``verbose`` output verbosity level, higher number will increase verbosity level
        ``lib_name`` specifying a library name (can be used multiple times to show multiple libraries in a single command) # noqa

        For more information see https://redis.io/commands/tfunction-list/
        """
        pieces = []
        if with_code:
            pieces.append('WITHCODE')
        if verbose >= 1 and verbose <= 3:
            pieces.append('v' * verbose)
        else:
            raise DataError('verbose can be 1, 2 or 3')
        if lib_name is not None:
            pieces.append('LIBRARY')
            pieces.append(lib_name)
        return self.execute_command('TFUNCTION LIST', *pieces)

    def _tfcall(self, lib_name: str, func_name: str, keys: KeysT=None, _async: bool=False, *args: List) -> ResponseT:
        pieces = [f'{lib_name}.{func_name}']
        if keys is not None:
            pieces.append(len(keys))
            pieces.extend(keys)
        else:
            pieces.append(0)
        if args is not None:
            pieces.extend(args)
        if _async:
            return self.execute_command('TFCALLASYNC', *pieces)
        return self.execute_command('TFCALL', *pieces)

    def tfcall(self, lib_name: str, func_name: str, keys: KeysT=None, *args: List) -> ResponseT:
        """
        Invoke a function.

        ``lib_name`` - the library name contains the function.
        ``func_name`` - the function name to run.
        ``keys`` - the keys that will be touched by the function.
        ``args`` - Additional argument to pass to the function.

        For more information see https://redis.io/commands/tfcall/
        """
        return self._tfcall(lib_name, func_name, keys, False, *args)

    def tfcall_async(self, lib_name: str, func_name: str, keys: KeysT=None, *args: List) -> ResponseT:
        """
        Invoke an async function (coroutine).

        ``lib_name`` - the library name contains the function.
        ``func_name`` - the function name to run.
        ``keys`` - the keys that will be touched by the function.
        ``args`` - Additional argument to pass to the function.

        For more information see https://redis.io/commands/tfcall/
        """
        return self._tfcall(lib_name, func_name, keys, True, *args)