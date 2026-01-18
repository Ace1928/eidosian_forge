import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_object_init_error():

    class TestCls(object):

        def __init__(self, **kwargs):
            super(TestCls, self).__init__(**kwargs)
    with pytest.raises(TypeError) as cm:
        TestCls(name='foo')
    assert str(cm.value).startswith('object.__init__() takes')