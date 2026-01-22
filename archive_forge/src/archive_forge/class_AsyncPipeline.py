import redis
from ...asyncio.client import Pipeline as AsyncioPipeline
from .commands import (
class AsyncPipeline(AsyncSearchCommands, AsyncioPipeline, Pipeline):
    """AsyncPipeline for the module."""