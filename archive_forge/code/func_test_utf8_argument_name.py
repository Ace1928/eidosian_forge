import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_utf8_argument_name():
    c = rinterface.globalenv.find('c')
    d = dict([(u'哈哈', 1)])
    res = c(**d)
    assert u'哈哈' == res.do_slot('names')[0]