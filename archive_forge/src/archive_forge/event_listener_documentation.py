import asyncio
from ray.util.annotations import PublicAPI
from ray.workflow.common import Event
import time
from typing import Callable
Optional. Called after an event has been checkpointed and a transaction can
        be safely committed.