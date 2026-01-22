from __future__ import annotations
import logging  # isort:skip
from sphinx.ext.autodoc import (
from bokeh.colors.color import Color
from bokeh.core.enums import Enumeration
from bokeh.core.property.descriptors import PropertyDescriptor
from bokeh.model import Model
from . import PARALLEL_SAFE
class ColorDocumenter(ModuleLevelDocumenter):
    directivetype = 'bokeh-color'
    objtype = ''
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Color)

    def add_content(self, more_content, no_docstring=False):
        pass

    def get_object_members(self, want_all):
        return (False, [])