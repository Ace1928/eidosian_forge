import os
import ovs.util
import ovs.vlog
class Reconnect(object):
    name = 'RECONNECT'
    is_connected = False

    @staticmethod
    def deadline(fsm, now):
        return fsm.state_entered

    @staticmethod
    def run(fsm, now):
        return DISCONNECT