import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class AsyncScript:
    """
    An executable Lua script object returned by ``register_script``
    """

    def __init__(self, registered_client: 'AsyncRedis', script: ScriptTextT):
        self.registered_client = registered_client
        self.script = script
        if isinstance(script, str):
            try:
                encoder = registered_client.connection_pool.get_encoder()
            except AttributeError:
                encoder = registered_client.get_encoder()
            script = encoder.encode(script)
        self.sha = hashlib.sha1(script).hexdigest()

    async def __call__(self, keys: Union[Sequence[KeyT], None]=None, args: Union[Iterable[EncodableT], None]=None, client: Union['AsyncRedis', None]=None):
        """Execute the script, passing any required ``args``"""
        keys = keys or []
        args = args or []
        if client is None:
            client = self.registered_client
        args = tuple(keys) + tuple(args)
        from redis.asyncio.client import Pipeline
        if isinstance(client, Pipeline):
            client.scripts.add(self)
        try:
            return await client.evalsha(self.sha, len(keys), *args)
        except NoScriptError:
            self.sha = await client.script_load(self.script)
            return await client.evalsha(self.sha, len(keys), *args)