import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
class ExportedProgramSerializer:

    def __init__(self, opset_version: Optional[Dict[str, int]]=None):
        self.opset_version: Dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        if 'aten' not in self.opset_version:
            self.opset_version['aten'] = torch._C._get_max_operator_version()

    def serialize(self, exported_program: ep.ExportedProgram) -> SerializedArtifact:
        """
        Args:
            exported_program: Exported Program to serialize
        """
        gm_serializer = GraphModuleSerializer(exported_program.graph_signature, exported_program.module_call_graph)
        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)
        serialized_range_constraints = serialize_range_constraints(exported_program.range_constraints)
        constants = {}
        for n, c in gm_serializer.custom_objs.items():
            constants[n] = c
        for n, t in exported_program.tensor_constants.items():
            assert n not in constants
            constants[n] = t
        return SerializedArtifact(ExportedProgram(graph_module=serialized_graph_module, opset_version=self.opset_version, range_constraints=serialized_range_constraints, schema_version=SCHEMA_VERSION, dialect=exported_program.dialect), serialize_torch_artifact(exported_program.state_dict), serialize_torch_artifact(constants))