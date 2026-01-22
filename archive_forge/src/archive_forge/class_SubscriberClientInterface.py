from abc import abstractmethod, ABCMeta
from typing import (
from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.types import (
class SubscriberClientInterface(ContextManager, metaclass=ABCMeta):
    """
    A SubscriberClientInterface reads messages similar to Google Pub/Sub.
    Any subscribe failures are unlikely to succeed if retried.

    Must be used in a `with` block or have __enter__() called before use.
    """

    @abstractmethod
    def subscribe(self, subscription: Union[SubscriptionPath, str], callback: MessageCallback, per_partition_flow_control_settings: FlowControlSettings, fixed_partitions: Optional[Set[Partition]]=None) -> StreamingPullFuture:
        """
        This method starts a background thread to begin pulling messages from
        a Pub/Sub Lite subscription and scheduling them to be processed using the
        provided ``callback``.

        Args:
          subscription: The subscription to subscribe to.
          callback: The callback function. This function receives the message as its only argument.
          per_partition_flow_control_settings: The flow control settings for each partition subscribed to. Note that these
              settings apply to each partition individually, not in aggregate.
          fixed_partitions: A fixed set of partitions to subscribe to. If not present, will instead use auto-assignment.

        Returns:
          A StreamingPullFuture instance that can be used to manage the background stream.

        Raises:
          GoogleApiCallError: On a permanent failure.
        """
        raise NotImplementedError()