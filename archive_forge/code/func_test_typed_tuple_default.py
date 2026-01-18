import array
import datetime as dt
import pytest
from unittest import TestCase
from traitlets import HasTraits, Int, TraitError
from traitlets.tests.test_traitlets import TraitTestBase
from ipywidgets import Color, NumberFormat
from ipywidgets.widgets.widget import _remove_buffers, _put_buffers
from ipywidgets.widgets.trait_types import date_serialization, TypedTuple
def test_typed_tuple_default():

    class TestCase(HasTraits):
        value = TypedTuple(default_value=(1, 2, 3))
    obj = TestCase()
    assert obj.value == (1, 2, 3)