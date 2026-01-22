from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkValueValuesEnum(_messages.Enum):
    """Immutable. The Ethereum environment being accessed.

    Values:
      NETWORK_UNSPECIFIED: The network has not been specified, but should be.
      MAINNET: The Ethereum Mainnet.
      TESTNET_GOERLI_PRATER: Deprecated: The Ethereum Testnet based on Goerli
        protocol. Please use another test network.
      TESTNET_SEPOLIA: The Ethereum Testnet based on Sepolia/Bepolia protocol.
        See https://github.com/eth-clients/sepolia.
      TESTNET_HOLESKY: The Ethereum Testnet based on Holesky specification.
        See https://github.com/eth-clients/holesky.
    """
    NETWORK_UNSPECIFIED = 0
    MAINNET = 1
    TESTNET_GOERLI_PRATER = 2
    TESTNET_SEPOLIA = 3
    TESTNET_HOLESKY = 4