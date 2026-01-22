import csv
import json
import logging
def _handle_corrupt_row(self, message):
    """Handle corrupt rows.

    Depending on whether the decoder is configured to fail on error it will
    raise a DecodeError or return None.

    Args:
      message: String, the error message to raise.
    Returns:
      None, when the decoder is not configured to fail on error.
    Raises:
      DecodeError: when the decoder is configured to fail on error.
    """
    if self._fail_on_error:
        raise DecodeError(message)
    else:
        logging.warning('Discarding invalid row: %s', message)
        return None