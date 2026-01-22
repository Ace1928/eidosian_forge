from typing import Any
from cirq.testing.equals_tester import EqualsTester
class _LargerThanEverythingElse:

    def __eq__(self, other) -> bool:
        return isinstance(other, _LargerThanEverythingElse)

    def __ne__(self, other) -> bool:
        return not isinstance(other, _LargerThanEverythingElse)

    def __lt__(self, other) -> bool:
        return False

    def __le__(self, other) -> bool:
        return isinstance(other, _LargerThanEverythingElse)

    def __gt__(self, other) -> bool:
        return not isinstance(other, _LargerThanEverythingElse)

    def __ge__(self, other) -> bool:
        return True

    def __repr__(self) -> str:
        return 'LargerThanEverythingElse()'