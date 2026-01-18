import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_union_from_json_correct_widget_data():
    dummy = Widget()
    json_data = widget_serialization['to_json'](dummy, None)
    data = data_union_from_json(json_data, None)
    assert data == dummy