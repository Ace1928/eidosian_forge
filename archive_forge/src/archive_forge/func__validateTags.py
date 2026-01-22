import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def _validateTags(self, tags):
    """
        Checks the tag dict for errors and raises L{ValueError} if an
        error is found.

        @param tags: The tag dict passed to sendMsg.
        """
    for tag, value in tags.items():
        if not tag:
            raise ValueError('A tag name is required.')
        for char in tag:
            if not char.isalnum() and char not in ('-', '/', '.'):
                raise ValueError('Tag contains invalid characters.')