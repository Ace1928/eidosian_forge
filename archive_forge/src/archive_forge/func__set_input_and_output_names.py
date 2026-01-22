from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _set_input_and_output_names(graph, input_names, output_names):

    @_beartype.beartype
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError('number of %s names provided (%d) exceeded number of %ss (%d)' % (descriptor, len(name_list), descriptor, len(node_list)))
        output_node_set = set()
        for i, (name, node) in enumerate(zip(name_list, node_list)):
            if descriptor == 'output':
                if node in output_node_set:
                    identity_node = graph.create('onnx::Identity')
                    identity_node.insertAfter(node.node())
                    identity_node.addInput(node)
                    identity_node.output().setType(node.type())
                    graph.return_node().replaceInput(i, identity_node.output())
                    node = identity_node.output()
                output_node_set.add(node)
            if node.debugName() != name:
                node.setDebugName(name)
    set_names(list(graph.inputs()), input_names, 'input')
    set_names(list(graph.outputs()), output_names, 'output')