from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _parse_event(self, event):
    response_dict = event.to_response_dict()
    parsed_response = self._parser.parse(response_dict, self._output_shape)
    if response_dict['status_code'] == 200:
        return parsed_response
    else:
        raise EventStreamError(parsed_response, self._operation_name)