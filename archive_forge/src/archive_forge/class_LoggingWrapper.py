import logging
import os
from os_ken.lib import ip
class LoggingWrapper(object):

    def write(self, message):
        LOG.info(message.rstrip('\n'))