import unittest
import inspect
import threading
def toUpper(self, data):
    if self.__upper:
        self.__upper.receive(data)