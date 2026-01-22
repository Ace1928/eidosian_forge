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
@staticmethod
def add_syslog_handler(facility=None):
    global syslog_facility, syslog_handler
    if (not facility or facility == syslog_facility) and syslog_handler:
        return
    logger = logging.getLogger('syslog')
    if os.environ.get('OVS_SYSLOG_METHOD') == 'null':
        logger.disabled = True
        return
    if facility is None:
        facility = syslog_facility
    new_handler = logging.handlers.SysLogHandler(address='/dev/log', facility=facility)
    if syslog_handler:
        logger.removeHandler(syslog_handler)
    syslog_handler = new_handler
    syslog_facility = facility
    logger.addHandler(syslog_handler)
    return