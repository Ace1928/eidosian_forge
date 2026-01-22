import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
class DataWidget(SimpleWidget):
    d = Instance(DataInstance, args=()).tag(sync=True, to_json=mview_serializer, from_json=deserializer)