import numpy as np
from numba import njit
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.target_extension import (
from numba.core.retarget import BasicRetarget

This is a test for an error with ir_utils._max_label not being updated
correctly. As a result of the error, inline_closurecall will incorrectly
overwrite an existing label, resulting in code that creates an effect-free
infinite loop, which is an undefined behavior to LLVM. LLVM will then assume
the function can never execute and an alternative code path will be taken
in spite of the value of any conditional branch that is guarding the alternative
code path.
