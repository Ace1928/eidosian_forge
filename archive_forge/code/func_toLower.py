import unittest
import inspect
import threading
def toLower(self, data):
    self.lock.acquire()
    if self.__lower:
        self.__lower.send(data)
    self.lock.release()