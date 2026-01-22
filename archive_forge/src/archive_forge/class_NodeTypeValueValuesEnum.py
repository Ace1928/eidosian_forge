from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeTypeValueValuesEnum(_messages.Enum):
    """Immutable. The type of Ethereum node.

    Values:
      NODE_TYPE_UNSPECIFIED: Node type has not been specified, but should be.
      LIGHT: An Ethereum node that only downloads Ethereum block headers.
      FULL: Keeps a complete copy of the blockchain data, and contributes to
        the network by receiving, validating, and forwarding transactions.
      ARCHIVE: Holds the same data as full node as well as all of the
        blockchain's history state data dating back to the Genesis Block.
    """
    NODE_TYPE_UNSPECIFIED = 0
    LIGHT = 1
    FULL = 2
    ARCHIVE = 3