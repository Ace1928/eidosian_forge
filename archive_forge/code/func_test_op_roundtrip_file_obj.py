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
def test_op_roundtrip_file_obj(tmpdir):
    filename = f'{tmpdir}/op.json'
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    with open(filename, 'w+') as file:
        cirq.to_json(op1, file)
        assert os.path.exists(filename)
        file.seek(0)
        op2 = cirq.read_json(file)
        assert op1 == op2
    gzip_filename = f'{tmpdir}/op.gz'
    with open(gzip_filename, 'w+b') as gzip_file:
        cirq.to_json_gzip(op1, gzip_file)
        assert os.path.exists(gzip_filename)
        gzip_file.seek(0)
        op3 = cirq.read_json_gzip(gzip_file)
        assert op1 == op3