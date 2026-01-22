from ray._private import log  # isort: skip # noqa: F401
import logging
import os
import sys
from ray import _version  # noqa: E402
import ray._raylet  # noqa: E402
from ray._raylet import (  # noqa: E402,F401
from ray._private.state import (  # noqa: E402,F401
from ray._private.worker import (  # noqa: E402,F401
import ray.actor  # noqa: E402,F401
from ray.actor import method  # noqa: E402,F401
from ray.cross_language import java_function, java_actor_class  # noqa: E402,F401
from ray.runtime_context import get_runtime_context  # noqa: E402,F401
from ray import internal  # noqa: E402,F401
from ray import util  # noqa: E402,F401
from ray import _private  # noqa: E402,F401
from ray.client_builder import client, ClientBuilder  # noqa: E402,F401
from ray._private.auto_init_hook import wrap_auto_init_for_all_apis  # noqa: E402
class _DeprecationWrapper:

    def __init__(self, name, real_worker):
        self._name = name
        self._real_worker = real_worker
        self._warned = set()

    def __getattr__(self, attr):
        value = getattr(self._real_worker, attr)
        if attr not in self._warned:
            self._warned.add(attr)
            logger.warning(f'DeprecationWarning: `ray.{self._name}.{attr}` is a private attribute and access will be removed in a future Ray version.')
        return value