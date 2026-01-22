from __future__ import annotations
import datetime
import json
import re
import sys
from collections import namedtuple
from io import StringIO
from typing import TYPE_CHECKING
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core.structure import Molecule, Structure
class HistoryNode(namedtuple('HistoryNode', ['name', 'url', 'description'])):
    """A HistoryNode represents a step in the chain of events that lead to a
    Structure. HistoryNodes leave 'breadcrumbs' so that you can trace back how
    a Structure was created. For example, a HistoryNode might represent pulling
    a Structure from an external database such as the ICSD or CSD. Or, it might
    represent the application of a code (e.g. pymatgen) to the Structure, with
    a custom description of how that code was applied (e.g. a site removal
    Transformation was applied).

    A HistoryNode contains three fields:

    Attributes:
        name (str): The name of a code or resource that this Structure encountered in its history.
        url (str): The URL of that code/resource.
        description (dict): A free-form description of how the code/resource is related to the Structure.
    """
    __slots__ = ()

    def as_dict(self) -> dict[str, str]:
        """Returns: Dict."""
        return {'name': self.name, 'url': self.url, 'description': self.description}

    @classmethod
    def from_dict(cls, dct: dict[str, str]) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            HistoryNode
        """
        return cls(dct['name'], dct['url'], dct['description'])

    @classmethod
    def parse_history_node(cls, h_node) -> Self:
        """Parses a History Node object from either a dict or a tuple.

        Args:
            h_node: A dict with name/url/description fields or a 3-element tuple.

        Returns:
            HistoryNode
        """
        if isinstance(h_node, dict):
            return cls.from_dict(h_node)
        if len(h_node) != 3:
            raise ValueError(f'Invalid History node, should be dict or (name, version, description) tuple: {h_node}')
        return cls(h_node[0], h_node[1], h_node[2])