from __future__ import annotations
import typing as t
from ...completion import (
from ...host_configs import (
from ..compat import (
from ..argparsing.parsers import (
from .value_parsers import (
from .key_value_parsers import (
from .helpers import (
class ControllerParser(Parser):
    """Composite argument parser for the controller."""

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        namespace = ControllerConfig()
        state.set_namespace(namespace)
        parser = ControllerKeyValueParser()
        parser.parse(state)
        return namespace

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        return ControllerKeyValueParser().document(state)