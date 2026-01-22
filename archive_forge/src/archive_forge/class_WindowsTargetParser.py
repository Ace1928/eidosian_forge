from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class WindowsTargetParser(TargetsNamespaceParser, TypeParser):
    """Composite argument parser for a Windows target."""

    @property
    def allow_inventory(self) -> bool:
        """True if inventory is allowed, otherwise False."""
        return True

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of type names and type parsers."""
        return self.get_internal_parsers(state.root_namespace.targets)

    def get_stateless_parsers(self) -> dict[str, Parser]:
        """Return a dictionary of type names and type parsers."""
        return self.get_internal_parsers([])

    def get_internal_parsers(self, targets: list[WindowsConfig]) -> dict[str, Parser]:
        """Return a dictionary of type names and type parsers."""
        parsers: dict[str, Parser] = {}
        if self.allow_inventory and (not targets):
            parsers.update(inventory=WindowsInventoryParser())
        if not targets or not any((isinstance(target, WindowsInventoryConfig) for target in targets)):
            if get_ci_provider().supports_core_ci_auth():
                parsers.update(remote=WindowsRemoteParser())
        return parsers

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section = f'{self.option_name} options (choose one):'
        state.sections[section] = ''
        state.sections[section] = '\n'.join([f'  {name}:{parser.document(state)}' for name, parser in self.get_stateless_parsers().items()])
        return None