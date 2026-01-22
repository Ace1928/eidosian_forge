from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def _WarnIfUnsupportedDoubleWildcard(self):
    """Warn if ** use may lead to undefined results."""
    if not self.object_name:
        return
    delimiter_bounded_url = self.delim + self.object_name + self.delim
    split_url = delimiter_bounded_url.split('{delim}**{delim}'.format(delim=self.delim))
    removed_correct_double_wildcards_url_string = ''.join(split_url)
    if '**' in removed_correct_double_wildcards_url_string:
        sys.stderr.write('** behavior is undefined if directly preceeded or followed by with characters other than / in the cloud and {} locally.'.format(os.sep))