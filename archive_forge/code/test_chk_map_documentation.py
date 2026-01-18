from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
Check the result of iter_interesting_nodes.

        Note that we no longer care how many steps are taken, etc, just that
        the right contents are returned.

        :param records: A list of record keys that should be yielded
        :param items: A list of items (key,value) that should be yielded.
        