from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EthereumDetails(_messages.Message):
    """Ethereum-specific blockchain node details.

  Enums:
    ConsensusClientValueValuesEnum: Immutable. The consensus client.
    ExecutionClientValueValuesEnum: Immutable. The execution client
    NetworkValueValuesEnum: Immutable. The Ethereum environment being
      accessed.
    NodeTypeValueValuesEnum: Immutable. The type of Ethereum node.

  Fields:
    additionalEndpoints: Output only. Ethereum-specific endpoint information.
    apiEnableAdmin: Immutable. Enables JSON-RPC access to functions in the
      `admin` namespace. Defaults to `false`.
    apiEnableDebug: Immutable. Enables JSON-RPC access to functions in the
      `debug` namespace. Defaults to `false`.
    consensusClient: Immutable. The consensus client.
    executionClient: Immutable. The execution client
    gethDetails: Details for the Geth execution client.
    network: Immutable. The Ethereum environment being accessed.
    nodeType: Immutable. The type of Ethereum node.
    validatorConfig: Configuration for validator-related parameters on the
      beacon client, and for any GCP-managed validator client.
  """

    class ConsensusClientValueValuesEnum(_messages.Enum):
        """Immutable. The consensus client.

    Values:
      CONSENSUS_CLIENT_UNSPECIFIED: Consensus client has not been specified,
        but should be.
      LIGHTHOUSE: Consensus client implementation written in Rust, maintained
        by Sigma Prime. See [Lighthouse - Sigma
        Prime](https://lighthouse.sigmaprime.io/) for details.
      ERIGON_EMBEDDED_CONSENSUS_LAYER: Erigon's embedded consensus client
        embedded in the execution client. Note this option is not currently
        available when creating new blockchain nodes. See [Erigon on
        GitHub](https://github.com/ledgerwatch/erigon#embedded-consensus-
        layer) for details.
    """
        CONSENSUS_CLIENT_UNSPECIFIED = 0
        LIGHTHOUSE = 1
        ERIGON_EMBEDDED_CONSENSUS_LAYER = 2

    class ExecutionClientValueValuesEnum(_messages.Enum):
        """Immutable. The execution client

    Values:
      EXECUTION_CLIENT_UNSPECIFIED: Execution client has not been specified,
        but should be.
      GETH: Official Go implementation of the Ethereum protocol. See [go-
        ethereum](https://geth.ethereum.org/) for details.
      ERIGON: An implementation of Ethereum (execution client), on the
        efficiency frontier, written in Go. See [Erigon on
        GitHub](https://github.com/ledgerwatch/erigon) for details.
    """
        EXECUTION_CLIENT_UNSPECIFIED = 0
        GETH = 1
        ERIGON = 2

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
    additionalEndpoints = _messages.MessageField('EthereumEndpoints', 1)
    apiEnableAdmin = _messages.BooleanField(2)
    apiEnableDebug = _messages.BooleanField(3)
    consensusClient = _messages.EnumField('ConsensusClientValueValuesEnum', 4)
    executionClient = _messages.EnumField('ExecutionClientValueValuesEnum', 5)
    gethDetails = _messages.MessageField('GethDetails', 6)
    network = _messages.EnumField('NetworkValueValuesEnum', 7)
    nodeType = _messages.EnumField('NodeTypeValueValuesEnum', 8)
    validatorConfig = _messages.MessageField('ValidatorConfig', 9)