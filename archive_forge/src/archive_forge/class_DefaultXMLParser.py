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
class DefaultXMLParser(object):

    def __init__(self, session):
        """
        DOM Parser

        :param session: ssh session object
        """
        self._session = session
        self._parsing_pos10 = 0
        self.logger = SessionLoggerAdapter(logger, {'session': self._session})

    def parse(self, data):
        """
        parse incoming RPC response from networking device.

        :param data: incoming RPC data from device
        :return: None
        """
        if data:
            self._session._buffer.seek(0, os.SEEK_END)
            self._session._buffer.write(data)
            if self._session._base == NetconfBase.BASE_11:
                self._parse11()
            else:
                self._parse10()

    def _parse10(self):
        """Messages are delimited by MSG_DELIM. The buffer could have grown by
        a maximum of BUF_SIZE bytes everytime this method is called. Retains
        state across method calls and if a chunk has been read it will not be
        considered again."""
        self.logger.debug('parsing netconf v1.0')
        buf = self._session._buffer
        buf.seek(self._parsing_pos10)
        if MSG_DELIM in buf.read().decode('UTF-8'):
            buf.seek(0)
            msg, _, remaining = buf.read().decode('UTF-8').partition(MSG_DELIM)
            msg = msg.strip()
            if sys.version < '3':
                self._session._dispatch_message(msg.encode())
            else:
                self._session._dispatch_message(msg)
            self._session._buffer = StringIO()
            self._parsing_pos10 = 0
            if len(remaining.strip()) > 0:
                if type(self._session.parser) != DefaultXMLParser:
                    self.logger.debug('send remaining data to SAX parser')
                    self._session.parser.parse(remaining.encode())
                else:
                    self.logger.debug('Trying another round of parsing since there is still data')
                    self._session._buffer.write(remaining.encode())
                    self._parse10()
        else:
            self._parsing_pos10 = buf.tell() - MSG_DELIM_LEN
            if self._parsing_pos10 < 0:
                self._parsing_pos10 = 0

    def _parse11(self):
        """Messages are split into chunks. Chunks and messages are delimited
        by the regex #RE_NC11_DELIM defined earlier in this file. Each
        time we get called here either a chunk delimiter or an
        end-of-message delimiter should be found iff there is enough
        data. If there is not enough data, we will wait for more. If a
        delimiter is found in the wrong place, a #NetconfFramingError
        will be raised."""
        self.logger.debug('_parse11: starting')
        self._session._buffer.seek(0, os.SEEK_SET)
        data = self._session._buffer.getvalue()
        data_len = len(data)
        start = 0
        self.logger.debug('_parse11: working with buffer of %d bytes', data_len)
        while True and start < data_len:
            self.logger.debug('_parse11: matching from %d bytes from start of buffer', start)
            re_result = RE_NC11_DELIM.match(data[start:].decode('utf-8', errors='ignore'))
            if not re_result:
                self.logger.debug('_parse11: no delimiter found, buffer="%s"', data[start:].decode())
                break
            re_start = re_result.start()
            re_end = re_result.end()
            self.logger.debug('_parse11: regular expression start=%d, end=%d', re_start, re_end)
            if re_start != 0:
                raise NetconfFramingError('_parse11: delimiter not at start of match buffer', data[start:])
            if re_result.group(2):
                start += re_end
                message = ''.join(self._session._message_list)
                self._session._message_list = []
                self.logger.debug('_parse11: found end of message delimiter')
                self._session._dispatch_message(message)
                break
            elif re_result.group(1):
                self.logger.debug('_parse11: found chunk delimiter')
                digits = int(re_result.group(1))
                self.logger.debug('_parse11: chunk size %d bytes', digits)
                if data_len - start >= re_end + digits:
                    fragment = textify(data[start + re_end:start + re_end + digits])
                    self._session._message_list.append(fragment)
                    start += re_end + digits
                    self.logger.debug('_parse11: appending %d bytes', digits)
                    self.logger.debug('_parse11: fragment = "%s"', fragment)
                else:
                    start += re_start
                    self.logger.debug('_parse11: not enough data for chunk yet')
                    self.logger.debug('_parse11: setting start to %d', start)
                    break
        if start > 0:
            self.logger.debug('_parse11: saving back rest of message after %d bytes, original size %d', start, data_len)
            self._session._buffer = StringIO(data[start:])
            if start < data_len:
                self.logger.debug('_parse11: still have data, may have another full message!')
                self._parse11()
        self.logger.debug('_parse11: ending')