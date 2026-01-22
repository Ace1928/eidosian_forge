from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
@classmethod
def _defining_class(cls, trait: TraitType[t.Any, t.Any], classes: t.Sequence[type[HasTraits]]) -> type[Configurable]:
    """Get the class that defines a trait

        For reducing redundant help output in config files.
        Returns the current class if:
        - the trait is defined on this class, or
        - the class where it is defined would not be in the config file

        Parameters
        ----------
        trait : Trait
            The trait to look for
        classes : list
            The list of other classes to consider for redundancy.
            Will return `cls` even if it is not defined on `cls`
            if the defining class is not in `classes`.
        """
    defining_cls = cls
    assert trait.name is not None
    for parent in cls.mro():
        if issubclass(parent, Configurable) and parent in classes and (parent.class_own_traits(config=True).get(trait.name, None) is trait):
            defining_cls = parent
    return defining_cls