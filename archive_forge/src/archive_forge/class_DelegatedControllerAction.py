from __future__ import annotations
from .argparsing import (
from .parsers import (
class DelegatedControllerAction(CompositeAction):
    """Composite action parser for the controller when delegation is supported."""

    def create_parser(self) -> NamespaceParser:
        """Return a namespace parser to parse the argument associated with this action."""
        return DelegatedControllerParser()