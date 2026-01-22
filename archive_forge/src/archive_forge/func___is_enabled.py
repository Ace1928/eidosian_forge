import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
def __is_enabled(self, level):
    level = LEVELS.get(level.lower(), logging.DEBUG)
    for f, f_level in Vlog.__mfl[self.name].items():
        f_level = LEVELS.get(f_level, logging.CRITICAL)
        if level >= f_level:
            return True
    return False