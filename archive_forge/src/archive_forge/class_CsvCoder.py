import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class CsvCoder(beam.coders.Coder):
    """A coder to encode and decode CSV formatted data.
  """

    class _WriterWrapper(object):
        """A wrapper for csv.writer / csv.DictWriter to make it picklable."""

        def __init__(self, column_names, delimiter, decode_to_dict):
            self._state = (column_names, delimiter, decode_to_dict)
            self._buffer = cStringIO()
            if decode_to_dict:
                self._writer = csv.DictWriter(self._buffer, column_names, lineterminator='', delimiter=delimiter)
            else:
                self._writer = csv.writer(self._buffer, lineterminator='', delimiter=delimiter)

        def encode_record(self, record):
            self._writer.writerow(record)
            value = self._buffer.getvalue()
            self._buffer.seek(0)
            self._buffer.truncate(0)
            return value

        def __getstate__(self):
            return self._state

        def __setstate__(self, state):
            self.__init__(*state)

    def __init__(self, column_names, numeric_column_names, delimiter=',', decode_to_dict=True, fail_on_error=True, skip_initial_space=False):
        """Initializes CsvCoder.

    Args:
      column_names: Tuple of strings. Order must match the order in the file.
      numeric_column_names: Tuple of strings. Contains column names that are
          numeric. Every name in numeric_column_names must also be in
          column_names.
      delimiter: A one-character string used to separate fields.
      decode_to_dict: Boolean indicating whether the docoder should generate a
          dictionary instead of a raw sequence. True by default.
      fail_on_error: Whether to fail if a corrupt row is found. Default is True.
      skip_initial_space: When True, whitespace immediately following the
          delimiter is ignored when reading.
    """
        self._decoder = _decoders.CsvDecoder(column_names, numeric_column_names, delimiter, decode_to_dict, fail_on_error, skip_initial_space)
        self._encoder = self._WriterWrapper(column_names=column_names, delimiter=delimiter, decode_to_dict=decode_to_dict)

    def decode(self, csv_line):
        """Decode csv line into a python dict.

    Args:
      csv_line: String. One csv line from the file.

    Returns:
      Python dict where the keys are the column names from the file. The dict
      values are strings or numbers depending if a column name was listed in
      numeric_column_names. Missing string columns have the value '', while
      missing numeric columns have the value None. If there is an error in
      parsing csv_line, a python dict is returned where every value is '' or
      None.

    Raises:
      Exception: The number of columns to not match.
    """
        return self._decoder.decode(csv_line)

    def encode(self, python_data):
        """Encode python dict to a csv-formatted string.

    Args:
      python_data: A python collection, depending on the value of decode_to_dict
          it will be a python dictionary where the keys are the column names or
          a sequence.

    Returns:
      A csv-formatted string. The order of the columns is given by column_names.
    """
        return self._encoder.encode_record(python_data)