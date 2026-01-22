from typing import List
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.task import cooperate
from twisted.python import log
from twisted.python.reflect import safe_str

        @see: C{IPushProducer.stopProducing}
        