from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def FzSeparations_fz_separation_equivalent_outparams_fn(self, idx, dst_cs, prf, color_params):
    """
    Helper for out-params of class method fz_separations::ll_fz_separation_equivalent() [fz_separation_equivalent()].
    """
    dst_color = ll_fz_separation_equivalent(self.m_internal, idx, dst_cs.m_internal, prf.m_internal, color_params.internal())
    return dst_color