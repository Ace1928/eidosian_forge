from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def to_response_dict(self, status_code=200):
    message_type = self.headers.get(':message-type')
    if message_type == 'error' or message_type == 'exception':
        status_code = 400
    return {'status_code': status_code, 'headers': self.headers, 'body': self.payload}