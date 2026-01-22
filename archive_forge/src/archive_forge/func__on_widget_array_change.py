from functools import partial
import numpy as np
from traitlets import Union, Instance, Undefined, TraitError
from .serializers import data_union_serialization
from .traits import NDArray
from .widgets import NDArrayWidget, NDArrayBase, NDArraySource
def _on_widget_array_change(self, union, change):
    inst = change['owner']
    union._notify_trait(self.name, inst, inst)