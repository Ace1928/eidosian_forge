import io
import sys
import unittest
class Module1(object):

    @staticmethod
    def setUpModule():
        results.append('Module1.setUpModule')

    @staticmethod
    def tearDownModule():
        results.append('Module1.tearDownModule')