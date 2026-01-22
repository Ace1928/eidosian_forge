from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
class IncompatibleInventoryDelta(errors.BzrError):
    """The delta could not be deserialised because its contents conflict with
    the allow_versioned_root or allow_tree_references flags of the
    deserializer.
    """
    internal_error = False