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
def FzXml_fz_new_display_list_from_svg_xml_outparams_fn(self, xmldoc, base_uri, dir):
    """
    Helper for out-params of class method fz_xml::ll_fz_new_display_list_from_svg_xml() [fz_new_display_list_from_svg_xml()].
    """
    ret, w, h = ll_fz_new_display_list_from_svg_xml(self.m_internal, xmldoc.m_internal, base_uri, dir.m_internal)
    return (FzDisplayList(ret), w, h)