from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainNodeSourceValueValuesEnum(_messages.Enum):
    """Immutable. The source of the blockchain node for the validator
    configurations to be deployed to.

    Values:
      BLOCKCHAIN_NODE_SOURCE_UNSPECIFIED: Blockchain node source has not been
        specified, but should be.
      NEW_BLOCKCHAIN_NODE: Create a new blockchain node to deploy the
        validators to.
      EXISTING_BLOCKCHAIN_NODE: Deploying blockchain validators to an existing
        blockchain node, or to no node.
    """
    BLOCKCHAIN_NODE_SOURCE_UNSPECIFIED = 0
    NEW_BLOCKCHAIN_NODE = 1
    EXISTING_BLOCKCHAIN_NODE = 2