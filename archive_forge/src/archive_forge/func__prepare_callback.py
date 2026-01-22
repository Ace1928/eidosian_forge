import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _prepare_callback(self, test_info, target_list, verbose_str, short_str):
    """Append a _TestInfo to the given target list and sets a callback
        method to be called by stopTest method.
        """
    target_list.append(test_info)

    def callback():
        """This callback prints the test method outcome to the stream,
            as well as the elapsed time.
            """
        if not self.elapsed_times:
            self.start_time = self.stop_time = 0
        if self.showAll:
            self.stream.writeln('(%.3fs) %s' % (test_info.get_elapsed_time(), verbose_str))
        elif self.dots:
            self.stream.write(short_str)
    self.callback = callback