from __future__ import annotations
from typing import Callable
from .ruler import Ruler
from .rules_core import (
from .rules_core.state_core import StateCore
class ParserCore:

    def __init__(self) -> None:
        self.ruler = Ruler[RuleFuncCoreType]()
        for name, rule in _rules:
            self.ruler.push(name, rule)

    def process(self, state: StateCore) -> None:
        """Executes core chain rules."""
        for rule in self.ruler.getRules(''):
            rule(state)