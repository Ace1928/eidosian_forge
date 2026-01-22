from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import keys
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import mutation
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction as gs_transaction
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import type as gs_type
class ListSessionsRequest(proto.Message):
    """The request for
    [ListSessions][google.spanner.v1.Spanner.ListSessions].

    Attributes:
        database (str):
            Required. The database in which to list
            sessions.
        page_size (int):
            Number of sessions to be returned in the
            response. If 0 or less, defaults to the server's
            maximum allowed page size.
        page_token (str):
            If non-empty, ``page_token`` should contain a
            [next_page_token][google.spanner.v1.ListSessionsResponse.next_page_token]
            from a previous
            [ListSessionsResponse][google.spanner.v1.ListSessionsResponse].
        filter (str):
            An expression for filtering the results of the request.
            Filter rules are case insensitive. The fields eligible for
            filtering are:

            -  ``labels.key`` where key is the name of a label

            Some examples of using filters are:

            -  ``labels.env:*`` --> The session has the label "env".
            -  ``labels.env:dev`` --> The session has the label "env"
               and the value of the label contains the string "dev".
    """
    database: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)