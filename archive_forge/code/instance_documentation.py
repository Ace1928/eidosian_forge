from __future__ import annotations
import logging # isort:skip
import types
from importlib import import_module
from typing import (
from ..has_props import HasProps
from ..serialization import Serializable
from ._sphinx import model_link, property_link, register_type_link
from .bases import Init, Property
from .singletons import Undefined
 Provide a deferred initializer for Instance defaults.

    This is useful for Bokeh models with Instance properties that should have
    unique default values for every model instance. Using an InstanceDefault
    will afford better user-facing documentation than a lambda initializer.

    