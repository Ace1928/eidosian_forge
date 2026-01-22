import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _update_option(self, options, option, value=None):
    """Update option if exists else adds it and returns new options."""
    opts = [x.strip() for x in options.split(',')] if options else []
    pos = self._option_exists(options, option)
    if pos:
        opts.pop(pos - 1)
    opt = '%s=%s' % (option, value) if value else option
    opts.append(opt)
    return ','.join(opts) if len(opts) > 1 else opts[0]