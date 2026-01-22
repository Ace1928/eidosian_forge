import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
class ConverterState:
    """
    Stores some state that persists between recursions into the element tree
    """

    def __init__(self, diagram_kwargs: typing.Optional[dict]=None):
        self._element_diagram_states: Dict[int, ElementState] = {}
        self.diagrams: Dict[int, EditablePartial[NamedDiagram]] = {}
        self.unnamed_index: int = 1
        self.index: int = 0
        self.diagram_kwargs: dict = diagram_kwargs or {}
        self.extracted_diagram_names: Set[str] = set()

    def __setitem__(self, key: int, value: ElementState):
        self._element_diagram_states[key] = value

    def __getitem__(self, key: int) -> ElementState:
        return self._element_diagram_states[key]

    def __delitem__(self, key: int):
        del self._element_diagram_states[key]

    def __contains__(self, key: int):
        return key in self._element_diagram_states

    def generate_unnamed(self) -> int:
        """
        Generate a number used in the name of an otherwise unnamed diagram
        """
        self.unnamed_index += 1
        return self.unnamed_index

    def generate_index(self) -> int:
        """
        Generate a number used to index a diagram
        """
        self.index += 1
        return self.index

    def extract_into_diagram(self, el_id: int):
        """
        Used when we encounter the same token twice in the same tree. When this
        happens, we replace all instances of that token with a terminal, and
        create a new subdiagram for the token
        """
        position = self[el_id]
        if position.parent:
            ret = EditablePartial.from_call(railroad.NonTerminal, text=position.name)
            if 'item' in position.parent.kwargs:
                position.parent.kwargs['item'] = ret
            elif 'items' in position.parent.kwargs:
                position.parent.kwargs['items'][position.parent_index] = ret
        if position.converted.func == railroad.Group:
            content = position.converted.kwargs['item']
        else:
            content = position.converted
        self.diagrams[el_id] = EditablePartial.from_call(NamedDiagram, name=position.name, diagram=EditablePartial.from_call(railroad.Diagram, content, **self.diagram_kwargs), index=position.number)
        del self[el_id]