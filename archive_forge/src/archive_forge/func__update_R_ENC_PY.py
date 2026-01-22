import abc
import atexit
import contextlib
import contextvars
import csv
import enum
import functools
import inspect
import os
import math
import platform
import signal
import subprocess
import textwrap
import threading
import typing
import warnings
from typing import Union
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import rpy2.rinterface_lib.embedded as embedded
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
import rpy2.rinterface_lib.memorymanagement as memorymanagement
from rpy2.rinterface_lib import na_values
from rpy2.rinterface_lib.sexp import NULL
from rpy2.rinterface_lib.sexp import NULLType
import rpy2.rinterface_lib.bufferprotocol as bufferprotocol
from rpy2.rinterface_lib import sexp
from rpy2.rinterface_lib.sexp import CharSexp  # noqa: F401
from rpy2.rinterface_lib.sexp import RTYPES
from rpy2.rinterface_lib.sexp import SexpVector
from rpy2.rinterface_lib.sexp import StrSexpVector
from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import SexpEnvironment
from rpy2.rinterface_lib.sexp import unserialize  # noqa: F401
from rpy2.rinterface_lib.sexp import emptyenv
from rpy2.rinterface_lib.sexp import baseenv
from rpy2.rinterface_lib.sexp import globalenv
def _update_R_ENC_PY():
    conversion._R_ENC_PY[openrlib.rlib.CE_UTF8] = 'utf-8'
    l10n_info = tuple(baseenv['l10n_info']())
    if platform.system() == 'Windows':
        val_latin1 = 'cp{:d}'.format(l10n_info[3][0])
    else:
        val_latin1 = 'latin1'
    conversion._R_ENC_PY[openrlib.rlib.CE_LATIN1] = val_latin1
    if l10n_info[1]:
        val_native = conversion._R_ENC_PY[openrlib.rlib.CE_UTF8]
    else:
        val_native = val_latin1
    conversion._R_ENC_PY[openrlib.rlib.CE_NATIVE] = val_native
    conversion._R_ENC_PY[openrlib.rlib.CE_ANY] = 'utf-8'