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
def __process_msg(self):
    json = self.parser.finish()
    self.parser = None
    if isinstance(json, str):
        vlog.warn('%s: error parsing stream: %s' % (self.name, json))
        self.error(errno.EPROTO)
        return
    msg = Message.from_json(json)
    if not isinstance(msg, Message):
        vlog.warn('%s: received bad JSON-RPC message: %s' % (self.name, msg))
        self.error(errno.EPROTO)
        return
    self.__log_msg('received', msg)
    return msg