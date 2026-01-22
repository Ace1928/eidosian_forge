from pexpect import ExceptionPexpect, TIMEOUT, EOF, spawn
import time
import os
import sys
import re
class ExceptionPxssh(ExceptionPexpect):
    """Raised for pxssh exceptions.
    """