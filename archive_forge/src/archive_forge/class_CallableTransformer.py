from abc import ABC, abstractmethod
from typing import Callable
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite_v1 import SequencedMessage
class CallableTransformer(MessageTransformer):

    def transform(self, source: SequencedMessage) -> PubsubMessage:
        return transformer(source)