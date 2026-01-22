from __future__ import annotations
import logging  # isort:skip
from sphinx.ext.autodoc import (
from bokeh.colors.color import Color
from bokeh.core.enums import Enumeration
from bokeh.core.property.descriptors import PropertyDescriptor
from bokeh.model import Model
from . import PARALLEL_SAFE
 Required Sphinx extension setup function. 