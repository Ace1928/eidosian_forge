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
def test_resolver_cache_vs_not_yet_serializable(mod_spec: ModuleJsonTestSpec):
    resolver_cache_types = set([n for n, _ in mod_spec.get_resolver_cache_types()])
    common = set(mod_spec.not_yet_serializable) & resolver_cache_types
    assert len(common) == 0, f"Issue with the JSON config of {mod_spec.name}.\nTypes are listed in both {mod_spec.name}.json_resolver_cache.py and in the 'not_yet_serializable' list in {mod_spec.test_data_path}/spec.py: \n {common}"