from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
def transform_op_tree(root: OP_TREE, op_transformation: Callable[[Operation], OP_TREE]=lambda e: e, iter_transformation: Callable[[Iterable[OP_TREE]], OP_TREE]=lambda e: e, preserve_moments: bool=False) -> OP_TREE:
    """Maps transformation functions onto the nodes of an OP_TREE.

    Args:
        root: The operation or tree of operations to transform.
        op_transformation: How to transform the operations (i.e. leaves).
        iter_transformation: How to transform the iterables (i.e. internal
            nodes).
        preserve_moments: Whether to leave Moments alone. If True, the
            transformation functions will not be applied to Moments or the
            operations within them.

    Returns:
        A transformed operation tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, Operation):
        return op_transformation(root)
    if preserve_moments and isinstance(root, moment.Moment):
        return root
    if isinstance(root, Iterable) and (not isinstance(root, str)):
        return iter_transformation((transform_op_tree(subtree, op_transformation, iter_transformation, preserve_moments) for subtree in root))
    _bad_op_tree(root)