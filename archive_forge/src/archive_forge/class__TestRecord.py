import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class _TestRecord:
    """Representation of a test."""

    def __init__(self, id, tags, details, status, timestamps):
        self.id = id
        self.tags = tags
        self.details = details
        self.status = status
        self.timestamps = timestamps

    @classmethod
    def create(cls, test_id, timestamp):
        return cls(id=test_id, tags=set(), details={}, status='unknown', timestamps=(timestamp, None))

    def set(self, *args, **kwargs):
        if args:
            setattr(self, args[0], args[1])
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def transform(self, data, value):
        getattr(self, data[0])[data[1]] = value
        return self

    def to_dict(self):
        """Convert record into a "test dict".

        A "test dict" is a concept used in other parts of the code-base. It
        has the following keys:

        * id: the test id.
        * tags: The tags for the test. A set of unicode strings.
        * details: A dict of file attachments - ``testtools.content.Content``
          objects.
        * status: One of the StreamResult status codes (including inprogress)
          or 'unknown' (used if only file events for a test were received...)
        * timestamps: A pair of timestamps - the first one received with this
          test id, and the one in the event that triggered the notification.
          Hung tests have a None for the second end event. Timestamps are not
          compared - their ordering is purely order received in the stream.
        """
        return {'id': self.id, 'tags': self.tags, 'details': self.details, 'status': self.status, 'timestamps': list(self.timestamps)}

    def got_timestamp(self, timestamp):
        """Called when we receive a timestamp.

        This will always update the second element of the 'timestamps' tuple.
        It doesn't compare timestamps at all.
        """
        return self.set(timestamps=(self.timestamps[0], timestamp))

    def got_file(self, file_name, file_bytes, mime_type=None):
        """Called when we receive file information.

        ``mime_type`` is only used when this is the first time we've seen data
        from this file.
        """
        if file_name in self.details:
            case = self
        else:
            content_type = _make_content_type(mime_type)
            content_bytes = []
            case = self.transform(['details', file_name], Content(content_type, lambda: content_bytes))
        case.details[file_name].iter_bytes().append(file_bytes)
        return case

    def to_test_case(self):
        """Convert into a TestCase object.

        :return: A PlaceHolder test object.
        """
        global PlaceHolder
        if PlaceHolder is None:
            from testtools.testcase import PlaceHolder
        outcome = _status_map[self.status]
        return PlaceHolder(self.id, outcome=outcome, details=self.details, tags=self.tags, timestamps=self.timestamps)