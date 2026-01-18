from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_post_comment(self, event):
    if self.event.comment[0] is None:
        return False
    comment = event.comment[0]
    self.write_comment(comment)
    return True