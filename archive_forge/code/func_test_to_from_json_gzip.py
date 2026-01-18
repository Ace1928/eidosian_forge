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
def test_to_from_json_gzip():
    a, b = cirq.LineQubit.range(2)
    test_circuit = cirq.Circuit(cirq.H(a), cirq.CX(a, b))
    gzip_data = cirq.to_json_gzip(test_circuit)
    unzip_circuit = cirq.read_json_gzip(gzip_raw=gzip_data)
    assert test_circuit == unzip_circuit
    with pytest.raises(ValueError):
        _ = cirq.read_json_gzip(io.StringIO(), gzip_raw=gzip_data)
    with pytest.raises(ValueError):
        _ = cirq.read_json_gzip()