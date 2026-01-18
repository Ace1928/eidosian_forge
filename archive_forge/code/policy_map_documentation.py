from collections import deque
import threading
from typing import Dict, Set
import logging
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import PolicyID
from ray.util.annotations import PublicAPI
Writes the least-recently used policy's state to the Ray object store.

        Also closes the session - if applicable - of the stashed policy.

        Returns:
            The least-recently used policy, that just got removed from the cache.
        