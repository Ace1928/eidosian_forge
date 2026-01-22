from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class InheritedPseudoElementMatches:
    """
    Inherited pseudo element matches from pseudos of an ancestor node.
    """
    pseudo_elements: typing.List[PseudoElementMatches]

    def to_json(self):
        json = dict()
        json['pseudoElements'] = [i.to_json() for i in self.pseudo_elements]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(pseudo_elements=[PseudoElementMatches.from_json(i) for i in json['pseudoElements']])