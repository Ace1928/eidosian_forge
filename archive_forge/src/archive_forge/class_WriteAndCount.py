from __future__ import print_function
import boto
import time
import uuid
from threading import Thread
class WriteAndCount(object):
    """
    A file-like object that counts the number of characters written.
    """

    def __init__(self):
        self.size = 0

    def write(self, data):
        self.size += len(data)
        time.sleep(0)