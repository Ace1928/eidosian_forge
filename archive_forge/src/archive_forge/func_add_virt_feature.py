import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def add_virt_feature(self, virt_feature, parent):
    self.add_multiple_virt_features([virt_feature], parent)