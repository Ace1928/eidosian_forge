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
@property
def versionless_url_string(self):
    if self.IsProvider():
        return '%s://' % self.scheme
    elif self.IsBucket():
        return self.bucket_url_string
    return '%s://%s/%s' % (self.scheme, self.bucket_name, self.object_name)