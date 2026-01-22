import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any
import numpy as np
import pytest
import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info
def _assert_json_roundtrip(o, tmpdir):
    cirq.to_json_gzip(o, f'{tmpdir}/o.json')
    o2 = cirq.read_json_gzip(f'{tmpdir}/o.json')
    assert o == o2