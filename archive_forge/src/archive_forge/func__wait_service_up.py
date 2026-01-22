import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _wait_service_up(host, port, timeout):
    beg_time = time.time()
    while time.time() - beg_time < timeout:
        if is_port_in_use(host, port):
            return True
        time.sleep(1)
    return False