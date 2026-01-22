import re
from io import BytesIO
from .. import errors
def _state_expecting_body(self):
    if len(self._buffer) >= self._current_record_length:
        body_bytes = self._buffer[:self._current_record_length]
        self._buffer = self._buffer[self._current_record_length:]
        record = (self._current_record_names, body_bytes)
        self._parsed_records.append(record)
        self._reset_current_record()
        self._state_handler = self._state_expecting_record_type