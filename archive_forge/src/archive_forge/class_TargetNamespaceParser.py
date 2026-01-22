from __future__ import annotations
import abc
import typing as t
from ..argparsing.parsers import (
class TargetNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for target namespace parsers involving a single target."""

    @property
    def option_name(self) -> str:
        """The option name used for this parser."""
        return '--target'

    @property
    def dest(self) -> str:
        """The name of the attribute where the value should be stored."""
        return 'targets'

    @property
    def use_list(self) -> bool:
        """True if the destination is a list, otherwise False."""
        return True

    @property
    def limit_one(self) -> bool:
        """True if only one target is allowed, otherwise False."""
        return True