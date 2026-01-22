import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _get_op_resolver_id(op_resolver_type=OpResolverType.AUTO):
    """Get a integer identifier for the op resolver."""
    return {OpResolverType.AUTO: 1, OpResolverType.BUILTIN: 1, OpResolverType.BUILTIN_REF: 2, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES: 3}.get(op_resolver_type, None)