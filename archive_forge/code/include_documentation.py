from __future__ import annotations
import logging # isort:skip
from copy import copy
from typing import TypeVar
from ..has_props import HasProps
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
 Include "mix-in" property collection in a Bokeh model.

    See :ref:`bokeh.core.property_mixins` for more details.

    