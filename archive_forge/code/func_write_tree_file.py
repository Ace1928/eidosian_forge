from __future__ import (absolute_import, division, print_function)
import os
from ansible.constants import TREE_DIR
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
from ansible.utils.path import makedirs_safe, unfrackpath
def write_tree_file(self, hostname, buf):
    """ write something into treedir/hostname """
    buf = to_bytes(buf)
    try:
        makedirs_safe(self.tree)
    except (OSError, IOError) as e:
        self._display.warning(u'Unable to access or create the configured directory (%s): %s' % (to_text(self.tree), to_text(e)))
    try:
        path = to_bytes(os.path.join(self.tree, hostname))
        with open(path, 'wb+') as fd:
            fd.write(buf)
    except (OSError, IOError) as e:
        self._display.warning(u"Unable to write to %s's file: %s" % (hostname, to_text(e)))