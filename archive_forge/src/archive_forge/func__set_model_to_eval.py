from collections import OrderedDict
import contextlib
from typing import Dict, Any
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
import torch
from ._proto_graph import node_proto
@contextlib.contextmanager
def _set_model_to_eval(model):
    """Context manager to temporarily set the training mode of ``model`` to eval."""
    if not isinstance(model, torch.jit.ScriptFunction):
        originally_training = model.training
        model.train(False)
        try:
            yield
        finally:
            model.train(originally_training)
    else:
        try:
            yield
        finally:
            pass