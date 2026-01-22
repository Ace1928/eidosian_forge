import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
@staticmethod
def _feed_feature_table(consumer, feature_tuples):
    """Handle the feature table (list of tuples), passing data to the consumer (PRIVATE).

        Used by the parse_records() and parse() methods.
        """
    consumer.start_feature_table()
    for feature_key, location_string, qualifiers in feature_tuples:
        consumer.feature_key(feature_key)
        consumer.location(location_string)
        for q_key, q_value in qualifiers:
            if q_value is None:
                consumer.feature_qualifier(q_key, q_value)
            else:
                consumer.feature_qualifier(q_key, q_value.replace('\n', ' '))