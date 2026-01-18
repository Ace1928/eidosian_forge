import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_reraise():

    def get_next(tb):
        if six.PY3:
            return tb.tb_next.tb_next
        else:
            return tb.tb_next
    e = Exception('blah')
    try:
        raise e
    except Exception:
        tp, val, tb = sys.exc_info()
    try:
        six.reraise(tp, val, tb)
    except Exception:
        tp2, value2, tb2 = sys.exc_info()
        assert tp2 is Exception
        assert value2 is e
        assert tb is get_next(tb2)
    try:
        six.reraise(tp, val)
    except Exception:
        tp2, value2, tb2 = sys.exc_info()
        assert tp2 is Exception
        assert value2 is e
        assert tb2 is not tb
    try:
        six.reraise(tp, val, tb2)
    except Exception:
        tp2, value2, tb3 = sys.exc_info()
        assert tp2 is Exception
        assert value2 is e
        assert get_next(tb3) is tb2
    try:
        six.reraise(tp, None, tb)
    except Exception:
        tp2, value2, tb2 = sys.exc_info()
        assert tp2 is Exception
        assert value2 is not val
        assert isinstance(value2, Exception)
        assert tb is get_next(tb2)