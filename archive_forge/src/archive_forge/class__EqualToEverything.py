from typing import Any
from cirq.testing.equals_tester import EqualsTester
class _EqualToEverything:

    def __eq__(self, other) -> bool:
        return True

    def __ne__(self, other) -> bool:
        return False

    def __lt__(self, other) -> bool:
        return False

    def __le__(self, other) -> bool:
        return True

    def __gt__(self, other) -> bool:
        return False

    def __ge__(self, other) -> bool:
        return True

    def __repr__(self) -> str:
        return '_EqualToEverything'