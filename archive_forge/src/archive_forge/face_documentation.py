import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
Creates a StreamStreamMultiCallable for a stream-stream method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.

        Returns:
          A StreamStreamMultiCallable value for the named stream-stream method.
        