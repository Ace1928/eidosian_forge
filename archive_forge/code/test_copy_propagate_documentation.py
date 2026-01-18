from numba import jit, njit
from numba.core import types, ir, config, compiler
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.tests.support import IRPreservingTestPipeline
import numpy as np
import unittest
make sure Interpreter._remove_unused_temporaries() does not generate
        invalid code for rare chained assignment cases
        