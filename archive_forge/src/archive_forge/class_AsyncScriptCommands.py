import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class AsyncScriptCommands(ScriptCommands):

    async def script_debug(self, *args) -> None:
        return super().script_debug()

    def register_script(self: 'AsyncRedis', script: ScriptTextT) -> AsyncScript:
        """
        Register a Lua ``script`` specifying the ``keys`` it will touch.
        Returns a Script object that is callable and hides the complexity of
        deal with scripts, keys, and shas. This is the preferred way to work
        with Lua scripts.
        """
        return AsyncScript(self, script)