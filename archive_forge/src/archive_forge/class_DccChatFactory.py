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
class DccChatFactory(protocol.ClientFactory):
    protocol = DccChat
    noisy = False

    def __init__(self, client, queryData):
        self.client = client
        self.queryData = queryData

    def buildProtocol(self, addr):
        p = self.protocol(client=self.client, queryData=self.queryData)
        p.factory = self
        return p

    def clientConnectionFailed(self, unused_connector, unused_reason):
        self.client.dcc_sessions.remove(self)

    def clientConnectionLost(self, unused_connector, unused_reason):
        self.client.dcc_sessions.remove(self)