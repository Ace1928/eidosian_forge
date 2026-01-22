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
class HasOldJsonDict:
    __module__ = 'test.noncirq.namespace'

    def __eq__(self, other):
        return isinstance(other, HasOldJsonDict)

    def _json_dict_(self):
        return {'cirq_type': 'test.noncirq.namespace.HasOldJsonDict'}

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()