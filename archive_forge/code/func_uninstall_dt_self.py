from lxml import etree
import sys
import re
import doctest
def uninstall_dt_self(self):
    self.dt_self._DocTestRunner__record_outcome = self.prev_func