from typing import Dict
class RoutingSwapTag:
    """A 'cirq.TaggedOperation' tag indicated that the operation is an inserted SWAP.

    A RoutingSwapTag is meant to be used to distinguish SWAP operations that are inserted during
    a routing procedure and SWAP operations that are part of the original circuit before routing.
    """

    def __eq__(self, other):
        return isinstance(other, RoutingSwapTag)

    def __str__(self) -> str:
        return '<r>'

    def __repr__(self) -> str:
        return 'cirq.RoutingSwapTag()'

    def _json_dict_(self) -> Dict[str, str]:
        return {}

    def __hash__(self):
        return hash(RoutingSwapTag)