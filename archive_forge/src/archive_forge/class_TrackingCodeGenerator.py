import typing as t
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment: 'Environment') -> None:
        super().__init__(environment, '<introspection>', '<introspection>')
        self.undeclared_identifiers: t.Set[str] = set()

    def write(self, x: str) -> None:
        """Don't write."""

    def enter_frame(self, frame: Frame) -> None:
        """Remember all undeclared identifiers."""
        super().enter_frame(frame)
        for _, (action, param) in frame.symbols.loads.items():
            if action == 'resolve' and param not in self.environment.globals:
                self.undeclared_identifiers.add(param)