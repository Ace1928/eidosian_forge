from __future__ import annotations
from .argparsing import (
from .parsers import (
class OriginControllerAction(CompositeAction):
    """Composite action parser for the controller when the only option is `origin`."""

    def create_parser(self) -> NamespaceParser:
        """Return a namespace parser to parse the argument associated with this action."""
        return OriginControllerParser()