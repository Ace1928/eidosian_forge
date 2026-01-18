import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def truncate_deserializer(json_data, widget):
    return DataInstance(json_data['data'][:20].tobytes() if json_data else None)