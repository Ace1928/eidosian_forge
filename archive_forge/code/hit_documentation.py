from itertools import chain
from Bio.SearchIO._utils import allitems, optionalcascade, getattr_str
from ._base import _BaseSearchObject
from .hsp import HSP
Sort the HSP objects.

        :param key: sorting function
        :type key: callable, accepts HSP, returns key for sorting
        :param reverse: whether to reverse sorting results or no
        :type reverse: bool
        :param in_place: whether to do in-place sorting or no
        :type in_place: bool

        ``sort`` defaults to sorting in-place, to mimic Python's ``list.sort``
        method. If you set the ``in_place`` argument to False, it will treat
        return a new, sorted Hit object and keep the initial one unsorted

        