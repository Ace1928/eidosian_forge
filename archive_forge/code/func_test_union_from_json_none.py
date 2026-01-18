import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_union_from_json_none():
    assert data_union_from_json(None, None) is None