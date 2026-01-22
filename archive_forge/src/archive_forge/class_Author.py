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
class Author(namedtuple('Author', ['name', 'email'])):
    """An Author contains two fields: name and email. It is meant to represent
    the author of a Structure or the author of a code that was applied to a Structure.
    """
    __slots__ = ()

    def __str__(self):
        """String representation of an Author."""
        return f'{self.name} <{self.email}>'

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'name': self.name, 'email': self.email}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Author
        """
        return cls(dct['name'], dct['email'])

    @classmethod
    def parse_author(cls, author) -> Self:
        """Parses an Author object from either a String, dict, or tuple.

        Args:
            author: A String formatted as "NAME <email@domain.com>",
                (name, email) tuple, or a dict with name and email keys.

        Returns:
            An Author object.
        """
        if isinstance(author, str):
            m = re.match('\\s*(.*?)\\s*<(.*?@.*?)>\\s*', author)
            if not m or m.start() != 0 or m.end() != len(author):
                raise ValueError(f'Invalid author format! {author}')
            return cls(m.groups()[0], m.groups()[1])
        if isinstance(author, dict):
            return cls.from_dict(author)
        if len(author) != 2:
            raise ValueError(f'Invalid author, should be String or (name, email) tuple: {author}')
        return cls(author[0], author[1])