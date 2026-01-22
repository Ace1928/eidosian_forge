from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class labels_view:
    """
    Scale labels (incl. caption & title) for the plot
    """
    x: Optional[str] = None
    y: Optional[str] = None
    alpha: Optional[str] = None
    color: Optional[str] = None
    colour: Optional[str] = None
    fill: Optional[str] = None
    linetype: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[str] = None
    stroke: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    subtitle: Optional[str] = None

    def update(self, other: labels_view):
        """
        Update the labels with those in other
        """
        for name, value in other.iter_set_fields():
            setattr(self, name, value)

    def add_defaults(self, other: labels_view):
        """
        Update labels that are missing with those in other
        """
        for name, value in other.iter_set_fields():
            cur_value = getattr(self, name)
            if cur_value is None:
                setattr(self, name, value)

    def iterfields(self) -> Iterator[tuple[str, Optional[str]]]:
        """
        Return an iterator of all (field, value) pairs
        """
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def iter_set_fields(self) -> Iterator[tuple[str, str]]:
        """
        Return an iterator of (field, value) pairs of none None values
        """
        return ((k, v) for k, v in self.iterfields() if v is not None)

    def get(self, name: str, default: str) -> str:
        """
        Get label value, return default if value is None
        """
        value = getattr(self, name)
        return str(value) if value is not None else default

    def __contains__(self, name: str) -> bool:
        """
        Return True if name has been set (is not None)
        """
        return getattr(self, name) is not None

    def __repr__(self) -> str:
        """
        Representations without the None values
        """
        nv_pairs = ', '.join((f'{name}={repr(value)}' for name, value in self.iter_set_fields()))
        return f'{self.__class__.__name__}({nv_pairs})'