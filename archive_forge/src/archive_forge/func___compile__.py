from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
def __compile__(self) -> None:
    """Build rules lookup cache"""
    chains = {''}
    for rule in self.__rules__:
        if not rule.enabled:
            continue
        for name in rule.alt:
            chains.add(name)
    self.__cache__ = {}
    for chain in chains:
        self.__cache__[chain] = []
        for rule in self.__rules__:
            if not rule.enabled:
                continue
            if chain and chain not in rule.alt:
                continue
            self.__cache__[chain].append(rule.fn)