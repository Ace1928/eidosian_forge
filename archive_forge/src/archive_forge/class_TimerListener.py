import asyncio
from ray.util.annotations import PublicAPI
from ray.workflow.common import Event
import time
from typing import Callable
@PublicAPI(stability='alpha')
class TimerListener(EventListener):
    """
    A listener that produces an event at a given timestamp.
    """

    async def poll_for_event(self, timestamp):
        await asyncio.sleep(timestamp - time.time())