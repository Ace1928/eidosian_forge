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
def FzDisplayList_fz_search_display_list_outparams_fn(self, needle, hit_bbox, hit_max):
    """
    Helper for out-params of class method fz_display_list::ll_fz_search_display_list() [fz_search_display_list()].
    """
    ret, hit_mark = ll_fz_search_display_list(self.m_internal, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)