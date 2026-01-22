import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
class _ReferenceableProducerWrapper(pb.Referenceable):

    def __init__(self, producer):
        self.producer = producer

    def remote_resumeProducing(self):
        self.producer.resumeProducing()

    def remote_pauseProducing(self):
        self.producer.pauseProducing()

    def remote_stopProducing(self):
        self.producer.stopProducing()