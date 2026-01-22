import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class ExcludeFilter:
    """
    ExcludeFilter is an exclusion filter.
    """

    def __init__(self, arg_lst):
        """
        Examples:
          ef1 = ExcludeFilter(["/etc/rc.*", "/etc/hostname"])
          ef2 = ExcludeFilter("/my/path/exclude.lst")
          Where exclude.lst contains:
          /etc/rc.*
          /etc/hostname

        Note: it is not possible to exclude a file if its encapsulating
              directory is itself watched. See this issue for more details
              https://github.com/seb-m/pyinotify/issues/31

        @param arg_lst: is either a list of patterns or a filename from which
                        patterns will be loaded.
        @type arg_lst: list of str or str
        """
        if isinstance(arg_lst, str):
            lst = self._load_patterns_from_file(arg_lst)
        elif isinstance(arg_lst, list):
            lst = arg_lst
        else:
            raise TypeError
        self._lregex = []
        for regex in lst:
            self._lregex.append(re.compile(regex, re.UNICODE))

    def _load_patterns_from_file(self, filename):
        lst = []
        with open(filename, 'r') as file_obj:
            for line in file_obj.readlines():
                pattern = line.strip()
                if not pattern or pattern.startswith('#'):
                    continue
                lst.append(pattern)
        return lst

    def _match(self, regex, path):
        return regex.match(path) is not None

    def __call__(self, path):
        """
        @param path: Path to match against provided regexps.
        @type path: str
        @return: Return True if path has been matched and should
                 be excluded, False otherwise.
        @rtype: bool
        """
        for regex in self._lregex:
            if self._match(regex, path):
                return True
        return False