import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def all_messages(self):
    """
        Return a dictionary of all the messages of this validator, and
        any subvalidators if present.  Keys are message names, values
        may be a message or list of messages.  This is really just
        intended for documentation purposes, to show someone all the
        messages that a validator or compound validator (like Schemas)
        can produce.

        @@: Should this produce a more structured set of messages, so
        that messages could be unpacked into a rendered form to see
        the placement of all the messages?  Well, probably so.
        """
    msgs = self._messages.copy()
    for v in self.subvalidators():
        inner = v.all_messages()
        for key, msg in inner:
            if key in msgs:
                if msgs[key] == msg:
                    continue
                if isinstance(msgs[key], list):
                    msgs[key].append(msg)
                else:
                    msgs[key] = [msgs[key], msg]
            else:
                msgs[key] = msg
    return msgs