import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class Tagger(CopyStreamResult):

    def status(self, **kwargs):
        tags = kwargs.get('test_tags')
        if not tags:
            tags = set()
        tags.update(new_tags)
        tags.difference_update(gone_tags)
        if tags:
            kwargs['test_tags'] = tags
        else:
            kwargs['test_tags'] = None
        super(Tagger, self).status(**kwargs)