import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _end_current_block(self):
    if not self.current_block.is_terminated:
        tryblk = self.dfainfo.active_try_block
        if tryblk is not None:
            self._insert_exception_check()
    self._remove_unused_temporaries()
    self._insert_outgoing_phis()