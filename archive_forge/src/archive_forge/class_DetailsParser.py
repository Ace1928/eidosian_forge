from io import BytesIO
from testtools import content, content_type
from testtools.compat import _b
from subunit import chunked
class DetailsParser(object):
    """Base class/API reference for details parsing."""