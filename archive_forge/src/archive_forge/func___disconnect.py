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
def __disconnect(self):
    if self.rpc is not None:
        self.rpc.error(EOF)
        self.rpc.close()
        self.rpc = None
    elif self.stream is not None:
        self.stream.close()
        self.stream = None
    else:
        return
    self.seqno += 1
    self.pick_remote()