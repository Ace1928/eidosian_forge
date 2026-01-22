import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
Getting to some parts of the codes imply that the
    set of supported of languages has changed.  Modify the
    supported languages to simulate this future code change.