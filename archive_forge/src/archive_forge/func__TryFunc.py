from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def _TryFunc(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except shutil.Error as e:
        args = e.args[0][0]
        if args[2].startswith('[Errno 13]'):
            exceptions.reraise(PermissionsError(message=args[2], path=os.path.abspath(args[0])))
        raise
    except (OSError, IOError) as e:
        if e.errno == errno.EACCES:
            exceptions.reraise(PermissionsError(message=encoding.Decode(e.strerror), path=encoding.Decode(os.path.abspath(e.filename))))
        raise