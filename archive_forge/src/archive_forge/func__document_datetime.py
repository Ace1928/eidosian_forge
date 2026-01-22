import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_datetime(self, section, value, path):
    datetime_tuple = parse_timestamp(value).timetuple()
    datetime_str = str(datetime_tuple[0])
    for i in range(1, len(datetime_tuple)):
        datetime_str += ', ' + str(datetime_tuple[i])
    section.write('datetime(%s),' % datetime_str)