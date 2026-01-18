import asyncio
import enum
import logging
import math
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
import ray
from ray._private.utils import load_class
from ray.actor import ActorHandle
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.exceptions import RayActorError
from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.utils import JavaActorHandleProxy, MetricsPusher
from ray.serve.generated.serve_pb2 import DeploymentRoute
from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
from ray.serve.grpc_util import RayServegRPCContext
from ray.util import metrics
Shutdown router gracefully.

        The metrics_pusher needs to be shutdown separately.
        