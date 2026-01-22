from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class CsvListSerializer(ArgumentSerializer):

    def __init__(self, list_sep):
        self.list_sep = list_sep

    def serialize(self, value):
        """Serializes a list as a CSV string or unicode."""
        if six.PY2:
            output = io.BytesIO()
            writer = csv.writer(output, delimiter=self.list_sep)
            writer.writerow([unicode(x).encode('utf-8') for x in value])
            serialized_value = output.getvalue().decode('utf-8').strip()
        else:
            output = io.StringIO()
            writer = csv.writer(output, delimiter=self.list_sep)
            writer.writerow([str(x) for x in value])
            serialized_value = output.getvalue().strip()
        return _helpers.str_or_unicode(serialized_value)