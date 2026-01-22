import codecs
import errno
import os
import random
import sys
import ovs.json
import ovs.poller
import ovs.reconnect
import ovs.stream
import ovs.timeval
import ovs.util
import ovs.vlog
def __validate_arg(self, value, name, must_have):
    if (value is not None) == (must_have != 0):
        return None
    else:
        type_name = Message.type_to_string(self.type)
        if must_have:
            verb = 'must'
        else:
            verb = 'must not'
        return '%s %s have "%s"' % (type_name, verb, name)