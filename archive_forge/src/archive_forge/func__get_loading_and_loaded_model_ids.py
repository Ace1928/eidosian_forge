import asyncio
import inspect
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, List, Set
from ray._private.async_compat import sync_to_async
from ray.serve import metrics
from ray.serve._private.common import DeploymentID, MultiplexedReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import MetricsPusher
from ray.serve.context import _get_global_client, _get_internal_replica_context
def _get_loading_and_loaded_model_ids(self) -> List[str]:
    """Get the model IDs of the loaded models & loading models in the replica.
        This is to push the model id information early to the controller, so that
        requests can be routed to the replica.
        """
    models_list = set(self.models.keys())
    models_list.update(self._model_load_tasks)
    return list(models_list)