from enum import Enum
import itertools
import logging
import time
from rustworkx import vf2_mapping
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
run the layout method