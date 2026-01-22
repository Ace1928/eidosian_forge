import os
from threading import Lock
import difflib
from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.builder import E
from xml.sax._exceptions import SAXParseException
from xml.sax import make_parser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.operations import rpc
from ncclient.transport.parser import SAXFilterXMLNotFoundError
from ncclient.transport.parser import MSG_DELIM, MSG_DELIM_LEN
from ncclient.operations.errors import OperationError
import logging
from ncclient.xml_ import BASE_NS_1_0
def _delimiter_check(self, data):
    """
        SAX parser throws SAXParseException exception, if there is extra data
        after MSG_DELIM

        :param data: content read by select loop
        :return: None
        """
    data = data.decode('UTF-8')
    if MSG_DELIM in data:
        msg, delim, remaining = data.partition(MSG_DELIM)
        self._session._buffer.seek(0, os.SEEK_END)
        self._session._buffer.write(delim.encode())
        if remaining.strip() != '':
            self._session._buffer.write(remaining.encode())
        self.sax_parser = make_parser()
        self.sax_parser.setContentHandler(SAXParser(self._session))
    elif RPC_REPLY_END_TAG in data or RFC_RPC_REPLY_END_TAG in data:
        tag = RPC_REPLY_END_TAG if RPC_REPLY_END_TAG in data else RFC_RPC_REPLY_END_TAG
        logger.warning('Check for rpc reply end tag within data received: %s' % data)
        msg, delim, remaining = data.partition(tag)
        self._session._buffer.seek(0, os.SEEK_END)
        self._session._buffer.write(remaining.encode())
    else:
        logger.warning('Check if end delimiter is split within data received: %s' % data)
        buf = self._session._buffer
        buf.seek(buf.tell() - len(RFC_RPC_REPLY_END_TAG) - MSG_DELIM_LEN)
        rpc_response_last_msg = buf.read().decode('UTF-8').replace('\n', '')
        if RPC_REPLY_END_TAG in rpc_response_last_msg or RFC_RPC_REPLY_END_TAG in rpc_response_last_msg:
            tag = RPC_REPLY_END_TAG if RPC_REPLY_END_TAG in rpc_response_last_msg else RFC_RPC_REPLY_END_TAG
            match_obj = difflib.SequenceMatcher(None, rpc_response_last_msg, data).get_matching_blocks()
            if match_obj:
                if match_obj[0].b == 0:
                    self._delimiter_check((rpc_response_last_msg + data[match_obj[0].size:]).encode())
                else:
                    data = rpc_response_last_msg + data
                    if MSG_DELIM in data:
                        clean_up = len(rpc_response_last_msg) - (rpc_response_last_msg.find(tag) + len(tag))
                        self._session._buffer.truncate(buf.tell() - clean_up)
                        self._delimiter_check(data.encode())
                    else:
                        self._delimiter_check((rpc_response_last_msg + data).encode())