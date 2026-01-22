import enum
from typing import Optional, List, Union, Iterable, Tuple
class SwitchStatementPreview(Statement):
    """AST node for the proposed 'switch-case' extension to OpenQASM 3, before the syntax was
    stabilized.  This corresponds to the :attr:`.ExperimentalFeatures.SWITCH_CASE_V1` logic.

    The stabilized form of the syntax instead uses :class:`.SwitchStatement`."""

    def __init__(self, target: Expression, cases: Iterable[Tuple[Iterable[Expression], ProgramBlock]]):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]