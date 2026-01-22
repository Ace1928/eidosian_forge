import collections
import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI
import ray
from ray import cloudpickle
from ray._private.serialization import pickle_dumps
from ray.dag import DAGNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve._private.http_util import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import (
from ray.serve.context import (
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.multiplex import _ModelMultiplexWrapper
from ray.serve.schema import LoggingConfig, ServeInstanceDetails, ServeStatus
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.serve._private import api as _private_api  # isort:skip
class ASGIIngressWrapper(cls, ASGIAppReplicaWrapper):

    def __init__(self, *args, **kwargs):
        cls.__init__(self, *args, **kwargs)
        ServeUsageTag.FASTAPI_USED.record('1')
        ASGIAppReplicaWrapper.__init__(self, frozen_app)

    async def __del__(self):
        await ASGIAppReplicaWrapper.__del__(self)
        if hasattr(cls, '__del__'):
            if inspect.iscoroutinefunction(cls.__del__):
                await cls.__del__(self)
            else:
                cls.__del__(self)