import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def _is_single(obj):
    """Check whether `obj` is a single document or an entire corpus.

    Parameters
    ----------
    obj : object

    Return
    ------
    (bool, object)
        2-tuple ``(is_single_document, new_obj)`` tuple, where `new_obj`
        yields the same sequence as the original `obj`.

    Notes
    -----
    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.

    """
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = itertools.chain([peek], obj_iter)
    except StopIteration:
        return (True, obj)
    if isinstance(peek, str):
        return (True, obj_iter)
    if temp_iter is obj:
        return (False, obj_iter)
    return (False, obj)