import os
import sys
import re
from xml.sax.handler import ContentHandler
from ncclient.transport.errors import NetconfFramingError
from ncclient.transport.session import NetconfBase
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.operations.errors import OperationError
from ncclient.transport import SessionListener
import logging
class SAXParserHandler(SessionListener):

    def __init__(self, session):
        self._session = session

    def callback(self, root, raw):
        if type(self._session.parser) == DefaultXMLParser:
            self._session.parser = self._session._device_handler.get_xml_parser(self._session)

    def errback(self, _):
        pass