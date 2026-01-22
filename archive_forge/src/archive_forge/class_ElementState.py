import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
class ElementState:
    """
    State recorded for an individual pyparsing Element
    """

    def __init__(self, element: pyparsing.ParserElement, converted: EditablePartial, parent: EditablePartial, number: int, name: str=None, parent_index: typing.Optional[int]=None):
        self.element: pyparsing.ParserElement = element
        self.name: typing.Optional[str] = name
        self.converted: EditablePartial = converted
        self.parent: EditablePartial = parent
        self.number: int = number
        self.parent_index: typing.Optional[int] = parent_index
        self.extract: bool = False
        self.complete: bool = False

    def mark_for_extraction(self, el_id: int, state: 'ConverterState', name: str=None, force: bool=False):
        """
        Called when this instance has been seen twice, and thus should eventually be extracted into a sub-diagram
        :param el_id: id of the element
        :param state: element/diagram state tracker
        :param name: name to use for this element's text
        :param force: If true, force extraction now, regardless of the state of this. Only useful for extracting the
        root element when we know we're finished
        """
        self.extract = True
        if not self.name:
            if name:
                self.name = name
            elif self.element.customName:
                self.name = self.element.customName
            else:
                self.name = ''
        if force or (self.complete and _worth_extracting(self.element)):
            state.extract_into_diagram(el_id)