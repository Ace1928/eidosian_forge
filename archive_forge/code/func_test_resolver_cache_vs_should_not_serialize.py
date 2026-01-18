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
@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS, ids=repr)
def test_resolver_cache_vs_should_not_serialize(mod_spec: ModuleJsonTestSpec):
    resolver_cache_types = set([n for n, _ in mod_spec.get_resolver_cache_types()])
    common = set(mod_spec.should_not_be_serialized) & resolver_cache_types
    assert len(common) == 0, f'Defined in both {mod_spec.name} Resolver Cache and should not be serialized:{common}'