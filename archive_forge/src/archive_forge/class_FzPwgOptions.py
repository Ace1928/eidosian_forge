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
class FzPwgOptions(object):
    """ Wrapper class for struct `fz_pwg_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_pwg_options`.
        """
        _mupdf.FzPwgOptions_swiginit(self, _mupdf.new_FzPwgOptions(*args))

    def media_class(self):
        return _mupdf.FzPwgOptions_media_class(self)

    def media_color(self):
        return _mupdf.FzPwgOptions_media_color(self)

    def media_type(self):
        return _mupdf.FzPwgOptions_media_type(self)

    def output_type(self):
        return _mupdf.FzPwgOptions_output_type(self)

    def advance_distance(self):
        return _mupdf.FzPwgOptions_advance_distance(self)

    def advance_media(self):
        return _mupdf.FzPwgOptions_advance_media(self)

    def collate(self):
        return _mupdf.FzPwgOptions_collate(self)

    def cut_media(self):
        return _mupdf.FzPwgOptions_cut_media(self)

    def duplex(self):
        return _mupdf.FzPwgOptions_duplex(self)

    def insert_sheet(self):
        return _mupdf.FzPwgOptions_insert_sheet(self)

    def jog(self):
        return _mupdf.FzPwgOptions_jog(self)

    def leading_edge(self):
        return _mupdf.FzPwgOptions_leading_edge(self)

    def manual_feed(self):
        return _mupdf.FzPwgOptions_manual_feed(self)

    def media_position(self):
        return _mupdf.FzPwgOptions_media_position(self)

    def media_weight(self):
        return _mupdf.FzPwgOptions_media_weight(self)

    def mirror_print(self):
        return _mupdf.FzPwgOptions_mirror_print(self)

    def negative_print(self):
        return _mupdf.FzPwgOptions_negative_print(self)

    def num_copies(self):
        return _mupdf.FzPwgOptions_num_copies(self)

    def orientation(self):
        return _mupdf.FzPwgOptions_orientation(self)

    def output_face_up(self):
        return _mupdf.FzPwgOptions_output_face_up(self)

    def PageSize(self):
        return _mupdf.FzPwgOptions_PageSize(self)

    def separations(self):
        return _mupdf.FzPwgOptions_separations(self)

    def tray_switch(self):
        return _mupdf.FzPwgOptions_tray_switch(self)

    def tumble(self):
        return _mupdf.FzPwgOptions_tumble(self)

    def media_type_num(self):
        return _mupdf.FzPwgOptions_media_type_num(self)

    def compression(self):
        return _mupdf.FzPwgOptions_compression(self)

    def row_count(self):
        return _mupdf.FzPwgOptions_row_count(self)

    def row_feed(self):
        return _mupdf.FzPwgOptions_row_feed(self)

    def row_step(self):
        return _mupdf.FzPwgOptions_row_step(self)

    def rendering_intent(self):
        return _mupdf.FzPwgOptions_rendering_intent(self)

    def page_size_name(self):
        return _mupdf.FzPwgOptions_page_size_name(self)
    __swig_destroy__ = _mupdf.delete_FzPwgOptions
    m_internal = property(_mupdf.FzPwgOptions_m_internal_get, _mupdf.FzPwgOptions_m_internal_set)
    s_num_instances = property(_mupdf.FzPwgOptions_s_num_instances_get, _mupdf.FzPwgOptions_s_num_instances_set, doc=' Wrapped data is held by value.')

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzPwgOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPwgOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPwgOptions___ne__(self, rhs)