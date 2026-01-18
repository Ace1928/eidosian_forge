import sys
from subunit import tag_stream
A filter to change tags on a subunit stream.

subunit-tags foo -> adds foo
subunit-tags foo -bar -> adds foo and removes bar
