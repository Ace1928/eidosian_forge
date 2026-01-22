from os_ken.lib import hub
import logging
class EventletIOFactory(object):

    @staticmethod
    def create_custom_event():
        LOG.debug('Create CustomEvent called')
        return hub.Event()

    @staticmethod
    def create_looping_call(funct, *args, **kwargs):
        LOG.debug('create_looping_call called')
        return LoopingCall(funct, *args, **kwargs)