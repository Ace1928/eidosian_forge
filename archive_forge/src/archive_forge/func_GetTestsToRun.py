import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
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