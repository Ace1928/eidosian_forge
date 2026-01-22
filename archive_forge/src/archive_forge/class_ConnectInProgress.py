import os
import ovs.util
import ovs.vlog
class ConnectInProgress(object):
    name = 'CONNECTING'
    is_connected = False

    @staticmethod
    def deadline(fsm, now):
        return fsm.state_entered + max(1000, fsm.backoff)

    @staticmethod
    def run(fsm, now):
        return DISCONNECT