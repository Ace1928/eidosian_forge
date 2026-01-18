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
Closes the current log file. (This is useful on Windows, to ensure
        that a reference to the file is not kept by the daemon in case of
        detach.)