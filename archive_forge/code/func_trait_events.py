from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
@classmethod
def trait_events(cls: type[HasTraits], name: str | None=None) -> dict[str, EventHandler]:
    """Get a ``dict`` of all the event handlers of this class.

        Parameters
        ----------
        name : str (default: None)
            The name of a trait of this class. If name is ``None`` then all
            the event handlers of this class will be returned instead.

        Returns
        -------
        The event handlers associated with a trait name, or all event handlers.
        """
    events = {}
    for k, v in getmembers(cls):
        if isinstance(v, EventHandler):
            if name is None:
                events[k] = v
            elif name in v.trait_names:
                events[k] = v
            elif hasattr(v, 'tags'):
                if cls.trait_names(**v.tags):
                    events[k] = v
    return events