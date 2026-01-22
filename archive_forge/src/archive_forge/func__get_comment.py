import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _get_comment(self, path, comments):
    key = re.sub('^\\.', '', ''.join(path))
    if comments and key in comments:
        return '# ' + comments[key]
    else:
        return ''