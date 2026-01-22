from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
class LineLogger(basic.LineReceiver):
    tag = None
    stream = None
    delimiter = b'\n'
    service = None

    def lineReceived(self, line):
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError:
            line = repr(line)
        self.service.log.info('[{tag}] {line}', tag=self.tag, line=line, stream=self.stream)