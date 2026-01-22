import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK
import sys  # @Reimport
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache
def array_to_meta_xml(array, name, format):
    type = array.dtype.kind
    slice = name
    l = len(array.shape)
    if format == '%':
        if l > 2:
            slice += '[0]' * (l - 2)
            for r in range(l - 2):
                array = array[0]
        if type == 'f':
            format = '.5f'
        elif type == 'i' or type == 'u':
            format = 'd'
        else:
            format = 's'
    else:
        format = format.replace('%', '')
    l = len(array.shape)
    reslice = ''
    if l > 2:
        raise Exception('%s has more than 2 dimensions.' % slice)
    elif l == 1:
        is_row = array.flags['C_CONTIGUOUS']
        if is_row:
            rows = 1
            cols = min(len(array), MAX_SLICE_SIZE)
            if cols < len(array):
                reslice = '[0:%s]' % cols
            array = array[0:cols]
        else:
            cols = 1
            rows = min(len(array), MAX_SLICE_SIZE)
            if rows < len(array):
                reslice = '[0:%s]' % rows
            array = array[0:rows]
    elif l == 2:
        rows = min(array.shape[-2], MAX_SLICE_SIZE)
        cols = min(array.shape[-1], MAX_SLICE_SIZE)
        if cols < array.shape[-1] or rows < array.shape[-2]:
            reslice = '[0:%s, 0:%s]' % (rows, cols)
        array = array[0:rows, 0:cols]
    if not slice.endswith(reslice):
        slice += reslice
    bounds = (0, 0)
    if type in 'biufc':
        bounds = (array.min(), array.max())
    xml = '<array slice="%s" rows="%s" cols="%s" format="%s" type="%s" max="%s" min="%s"/>' % (slice, rows, cols, format, type, bounds[1], bounds[0])
    return (array, xml, rows, cols, format)