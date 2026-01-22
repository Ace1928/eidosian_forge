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
class FzStoryElementPosition(object):
    """ Wrapper class for struct `fz_story_element_position`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_story_element_position`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_story_element_position`.
        """
        _mupdf.FzStoryElementPosition_swiginit(self, _mupdf.new_FzStoryElementPosition(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzStoryElementPosition_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzStoryElementPosition
    depth = property(_mupdf.FzStoryElementPosition_depth_get, _mupdf.FzStoryElementPosition_depth_set)
    heading = property(_mupdf.FzStoryElementPosition_heading_get, _mupdf.FzStoryElementPosition_heading_set)
    id = property(_mupdf.FzStoryElementPosition_id_get, _mupdf.FzStoryElementPosition_id_set)
    href = property(_mupdf.FzStoryElementPosition_href_get, _mupdf.FzStoryElementPosition_href_set)
    rect = property(_mupdf.FzStoryElementPosition_rect_get, _mupdf.FzStoryElementPosition_rect_set)
    text = property(_mupdf.FzStoryElementPosition_text_get, _mupdf.FzStoryElementPosition_text_set)
    open_close = property(_mupdf.FzStoryElementPosition_open_close_get, _mupdf.FzStoryElementPosition_open_close_set)
    rectangle_num = property(_mupdf.FzStoryElementPosition_rectangle_num_get, _mupdf.FzStoryElementPosition_rectangle_num_set)
    s_num_instances = property(_mupdf.FzStoryElementPosition_s_num_instances_get, _mupdf.FzStoryElementPosition_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzStoryElementPosition_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzStoryElementPosition___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzStoryElementPosition___ne__(self, rhs)