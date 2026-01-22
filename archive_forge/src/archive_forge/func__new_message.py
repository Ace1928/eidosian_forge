import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def _new_message(self):
    if self._old_style_factory:
        msg = self._factory()
    else:
        msg = self._factory(policy=self.policy)
    if self._cur and self._cur.get_content_type() == 'multipart/digest':
        msg.set_default_type('message/rfc822')
    if self._msgstack:
        self._msgstack[-1].attach(msg)
    self._msgstack.append(msg)
    self._cur = msg
    self._last = msg