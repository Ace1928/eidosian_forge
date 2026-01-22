from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
class PrettyJsonFormatter(JsonFormatter):
    """Formats output in human-legible JSON."""

    def __unicode__(self):
        return json.dumps(self._table, separators=(', ', ': '), sort_keys=True, indent=2, ensure_ascii=False)