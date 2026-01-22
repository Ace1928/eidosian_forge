from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class DelegatedControllerParser(ControllerNamespaceParser, TypeParser):
    """Composite argument parser for the controller when delegation is supported."""

    def get_stateless_parsers(self) -> dict[str, Parser]:
        """Return a dictionary of type names and type parsers."""
        parsers: dict[str, Parser] = dict(origin=OriginParser(), docker=DockerParser(controller=True))
        if get_ci_provider().supports_core_ci_auth():
            parsers.update(remote=PosixRemoteParser(controller=True))
        return parsers

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section = '--controller options:'
        state.sections[section] = ''
        state.sections[section] = '\n'.join([f'  {name}:{parser.document(state)}' for name, parser in self.get_stateless_parsers().items()])
        return None