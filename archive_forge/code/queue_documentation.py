import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import ray
from ray.util.annotations import PublicAPI
Terminates the underlying QueueActor.

        All of the resources reserved by the queue will be released.

        Args:
            force: If True, forcefully kill the actor, causing an
                immediate failure. If False, graceful
                actor termination will be attempted first, before falling back
                to a forceful kill.
            grace_period_s: If force is False, how long in seconds to
                wait for graceful termination before falling back to
                forceful kill.
        