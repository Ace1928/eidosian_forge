from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _convert_models(self):
    models = []
    for model in self.models:
        state = getattr(model, self.model_attribute)
        model_def = dict(state=state.name if isinstance(state, Enum) else state)
        model_def['name'] = model.name if hasattr(model, 'name') else str(id(model))
        model_def['class-name'] = 'self' if model == self else model.__module__ + '.' + model.__class__.__name__
        models.append(model_def)
    return models