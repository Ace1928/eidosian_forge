import logging
from typing import List, Optional, Union
from google.api_core.exceptions import InvalidArgument
from google.api_core.operation import Operation
from cloudsdk.google.protobuf.field_mask_pb2 import FieldMask  # pytype: disable=pyi-error
from google.cloud.pubsublite.admin_client_interface import AdminClientInterface
from google.cloud.pubsublite.types import (
from google.cloud.pubsublite.types.paths import ReservationPath
from google.cloud.pubsublite_v1 import (
def update_topic(self, topic: Topic, update_mask: FieldMask) -> Topic:
    return self._underlying.update_topic(topic=topic, update_mask=update_mask)