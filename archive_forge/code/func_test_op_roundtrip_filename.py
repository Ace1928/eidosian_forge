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
def test_op_roundtrip_filename(tmpdir):
    filename = f'{tmpdir}/op.json'
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    cirq.to_json(op1, filename)
    assert os.path.exists(filename)
    op2 = cirq.read_json(filename)
    assert op1 == op2
    gzip_filename = f'{tmpdir}/op.gz'
    cirq.to_json_gzip(op1, gzip_filename)
    assert os.path.exists(gzip_filename)
    op3 = cirq.read_json_gzip(gzip_filename)
    assert op1 == op3