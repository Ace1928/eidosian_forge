from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
def _unified_records(self):
    """Returns a list of unified records."""
    merged_records = dict()
    for identifier, records in self._id_map.items():
        if len(records) > 1:
            merged = records[0].copy()
            for record in records[1:]:
                merged.add_attributes(record.attributes)
            for record in records:
                merged_records[record] = merged
    if not merged_records:
        return list(self._records)
    added_merged_records = set()
    unified_records = list()
    for record in self._records:
        if record in merged_records:
            merged = merged_records[record]
            if merged not in added_merged_records:
                unified_records.append(merged)
                added_merged_records.add(merged)
        else:
            unified_records.append(record)
    return unified_records