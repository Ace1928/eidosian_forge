import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class DenseGeneratorData:
    """Internal helper class to allow for different matrix types without
    making the code a huge collection of if statements."""

    def decode_rows(self, stream, conversors):
        for row in stream:
            values = _parse_values(row)
            if isinstance(values, dict):
                if values and max(values) >= len(conversors):
                    raise BadDataFormat(row)
                values = [values[i] if i in values else 0 for i in range(len(conversors))]
            elif len(values) != len(conversors):
                raise BadDataFormat(row)
            yield self._decode_values(values, conversors)

    @staticmethod
    def _decode_values(values, conversors):
        try:
            values = [None if value is None else conversor(value) for conversor, value in zip(conversors, values)]
        except ValueError as exc:
            if 'float: ' in str(exc):
                raise BadNumericalValue()
        return values

    def encode_data(self, data, attributes):
        """(INTERNAL) Encodes a line of data.

        Data instances follow the csv format, i.e, attribute values are
        delimited by commas. After converted from csv.

        :param data: a list of values.
        :param attributes: a list of attributes. Used to check if data is valid.
        :return: a string with the encoded data line.
        """
        current_row = 0
        for inst in data:
            if len(inst) != len(attributes):
                raise BadObject('Instance %d has %d attributes, expected %d' % (current_row, len(inst), len(attributes)))
            new_data = []
            for value in inst:
                if value is None or value == '' or value != value:
                    s = '?'
                else:
                    s = encode_string(str(value))
                new_data.append(s)
            current_row += 1
            yield ','.join(new_data)