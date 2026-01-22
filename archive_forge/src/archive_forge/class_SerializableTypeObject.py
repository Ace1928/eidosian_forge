import contextlib
import dataclasses
import datetime
import importlib
import io
import json
import os
import pathlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type
from unittest import mock
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sympy
import cirq
from cirq._compat import proper_eq
from cirq.protocols import json_serialization
from cirq.testing.json import ModuleJsonTestSpec, spec_for, assert_json_roundtrip_works
@dataclasses.dataclass
class SerializableTypeObject:
    test_type: Type

    def _json_dict_(self):
        return {'test_type': json_serialization.json_cirq_type(self.test_type)}

    @classmethod
    def _from_json_dict_(cls, test_type, **kwargs):
        return cls(json_serialization.cirq_type_from_json(test_type))