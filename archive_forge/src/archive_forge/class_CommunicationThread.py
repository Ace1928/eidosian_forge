import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
class CommunicationThread(threading.Thread):

    def __init__(self, tests_queue):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = tests_queue
        self.finished = False
        from _pydev_bundle.pydev_imports import SimpleXMLRPCServer
        from _pydev_bundle import pydev_localhost
        server = SimpleXMLRPCServer((pydev_localhost.get_localhost(), 0), logRequests=False)
        server.register_function(self.GetTestsToRun)
        server.register_function(self.notifyStartTest)
        server.register_function(self.notifyTest)
        server.register_function(self.notifyCommands)
        self.port = server.socket.getsockname()[1]
        self.server = server

    def GetTestsToRun(self, job_id):
        """
        @param job_id:

        @return: list(str)
            Each entry is a string in the format: filename|Test.testName
        """
        try:
            ret = self.queue.get(block=False)
            return ret
        except:
            self.finished = True
            return []

    def notifyCommands(self, job_id, commands):
        for command in commands:
            getattr(self, command[0])(job_id, *command[1], **command[2])
        return True

    def notifyStartTest(self, job_id, *args, **kwargs):
        pydev_runfiles_xml_rpc.notifyStartTest(*args, **kwargs)
        return True

    def notifyTest(self, job_id, *args, **kwargs):
        pydev_runfiles_xml_rpc.notifyTest(*args, **kwargs)
        return True

    def shutdown(self):
        if hasattr(self.server, 'shutdown'):
            self.server.shutdown()
        else:
            self._shutdown = True

    def run(self):
        if hasattr(self.server, 'shutdown'):
            self.server.serve_forever()
        else:
            self._shutdown = False
            while not self._shutdown:
                self.server.handle_request()