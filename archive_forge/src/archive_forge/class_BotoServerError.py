import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class BotoServerError(StandardError):

    def __init__(self, status, reason, body=None, *args):
        super(BotoServerError, self).__init__(status, reason, body, *args)
        self.status = status
        self.reason = reason
        self.body = body or ''
        self.request_id = None
        self.error_code = None
        self._error_message = None
        self.message = ''
        self.box_usage = None
        if isinstance(self.body, bytes):
            try:
                self.body = self.body.decode('utf-8')
            except UnicodeDecodeError:
                boto.log.debug('Unable to decode body from bytes!')
        if self.body:
            if hasattr(self.body, 'items'):
                self.request_id = self.body.get('RequestId', None)
                if 'Error' in self.body:
                    error = self.body.get('Error', {})
                    self.error_code = error.get('Code', None)
                    self.message = error.get('Message', None)
                else:
                    self.message = self.body.get('message', None)
            else:
                try:
                    h = handler.XmlHandlerWrapper(self, self)
                    h.parseString(self.body)
                except (TypeError, xml.sax.SAXParseException):
                    try:
                        parsed = json.loads(self.body)
                        if 'RequestId' in parsed:
                            self.request_id = parsed['RequestId']
                        if 'Error' in parsed:
                            if 'Code' in parsed['Error']:
                                self.error_code = parsed['Error']['Code']
                            if 'Message' in parsed['Error']:
                                self.message = parsed['Error']['Message']
                    except (TypeError, ValueError):
                        self.message = self.body
                        self.body = None

    def __getattr__(self, name):
        if name == 'error_message':
            return self.message
        if name == 'code':
            return self.error_code
        raise AttributeError

    def __setattr__(self, name, value):
        if name == 'error_message':
            self.message = value
        else:
            super(BotoServerError, self).__setattr__(name, value)

    def __repr__(self):
        return '%s: %s %s\n%s' % (self.__class__.__name__, self.status, self.reason, self.body)

    def __str__(self):
        return '%s: %s %s\n%s' % (self.__class__.__name__, self.status, self.reason, self.body)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name in ('RequestId', 'RequestID'):
            self.request_id = value
        elif name == 'Code':
            self.error_code = value
        elif name == 'Message':
            self.message = value
        elif name == 'BoxUsage':
            self.box_usage = value
        return None

    def _cleanupParsedProperties(self):
        self.request_id = None
        self.error_code = None
        self.message = None
        self.box_usage = None