from collections import OrderedDict
import contextlib
from typing import Dict, Any
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
import torch
from ._proto_graph import node_proto
class NodePy(NodeBase):

    def __init__(self, node_cpp, valid_methods):
        super().__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []
        for m in valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)
                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)
            else:
                setattr(self, m, getattr(node_cpp, m)())