import contextlib
from dataclasses import dataclass
import logging
import os
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.workflow import common, workflow_storage
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from collections import ChainMap
import io
class ObjectRefPickler(cloudpickle.CloudPickler):
    _object_ref_reducer = {ray.ObjectRef: lambda ref: _reduce_objectref(workflow_id, ref, tasks)}
    dispatch_table = ChainMap(_object_ref_reducer, cloudpickle.CloudPickler.dispatch_table)
    dispatch = dispatch_table