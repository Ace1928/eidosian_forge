from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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