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
class FzTree(object):
    """
    Wrapper class for struct `fz_tree`. Not copyable or assignable.
    AA-tree to look up things by strings.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_new_tree_archive(self):
        """
        Class-aware wrapper for `::fz_new_tree_archive()`.
        	Create an archive that holds named buffers.

        	tree can either be a preformed tree with fz_buffers as values,
        	or it can be NULL for an empty tree.
        """
        return _mupdf.FzTree_fz_new_tree_archive(self)

    def fz_tree_lookup(self, key):
        """
        Class-aware wrapper for `::fz_tree_lookup()`.
        	Look for the value of a node in the tree with the given key.

        	Simple pointer equivalence is used for key.

        	Returns NULL for no match.
        """
        return _mupdf.FzTree_fz_tree_lookup(self, key)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_tree`.
        """
        _mupdf.FzTree_swiginit(self, _mupdf.new_FzTree(*args))
    __swig_destroy__ = _mupdf.delete_FzTree

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzTree_m_internal_value(self)
    m_internal = property(_mupdf.FzTree_m_internal_get, _mupdf.FzTree_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzTree_s_num_instances_get, _mupdf.FzTree_s_num_instances_set)