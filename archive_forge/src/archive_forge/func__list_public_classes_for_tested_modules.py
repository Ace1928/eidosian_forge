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
@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
def _list_public_classes_for_tested_modules():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sorted(((mod_spec, n, o) for mod_spec in MODULE_TEST_SPECS for n, o in mod_spec.find_classes_that_should_serialize()), key=lambda mno: (str(mno[0]), mno[1]))