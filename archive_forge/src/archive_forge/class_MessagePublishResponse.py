from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class MessagePublishResponse(proto.Message):
    """Response to a MessagePublishRequest.

    Attributes:
        start_cursor (google.cloud.pubsublite_v1.types.Cursor):
            The cursor of the first published message in
            the batch. The cursors for any remaining
            messages in the batch are guaranteed to be
            sequential.
        cursor_ranges (MutableSequence[google.cloud.pubsublite_v1.types.MessagePublishResponse.CursorRange]):
            Cursors for messages published in the batch.
            There will exist multiple ranges when cursors
            are not contiguous within the batch.
            The cursor ranges may not account for all
            messages in the batch when publish idempotency
            is enabled. A missing range indicates that
            cursors could not be determined for messages
            within the range, as they were deduplicated and
            the necessary data was not available at publish
            time. These messages will have offsets when
            received by a subscriber.
    """

    class CursorRange(proto.Message):
        """Cursors for a subrange of published messages.

        Attributes:
            start_cursor (google.cloud.pubsublite_v1.types.Cursor):
                The cursor of the message at the start index.
                The cursors for remaining messages up to the end
                index (exclusive) are sequential.
            start_index (int):
                Index of the message in the published batch
                that corresponds to the start cursor. Inclusive.
            end_index (int):
                Index of the last message in this range.
                Exclusive.
        """
        start_cursor: common.Cursor = proto.Field(proto.MESSAGE, number=1, message=common.Cursor)
        start_index: int = proto.Field(proto.INT32, number=2)
        end_index: int = proto.Field(proto.INT32, number=3)
    start_cursor: common.Cursor = proto.Field(proto.MESSAGE, number=1, message=common.Cursor)
    cursor_ranges: MutableSequence[CursorRange] = proto.RepeatedField(proto.MESSAGE, number=2, message=CursorRange)