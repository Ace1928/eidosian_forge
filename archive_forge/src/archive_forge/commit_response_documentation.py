from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
Additional statistics about a commit.

        Attributes:
            mutation_count (int):
                The total number of mutations for the transaction. Knowing
                the ``mutation_count`` value can help you maximize the
                number of mutations in a transaction and minimize the number
                of API round trips. You can also monitor this value to
                prevent transactions from exceeding the system
                `limit <https://cloud.google.com/spanner/quotas#limits_for_creating_reading_updating_and_deleting_data>`__.
                If the number of mutations exceeds the limit, the server
                returns
                `INVALID_ARGUMENT <https://cloud.google.com/spanner/docs/reference/rest/v1/Code#ENUM_VALUES.INVALID_ARGUMENT>`__.
        