import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _table_to_blocks(options, block_table, categories, extension_columns):
    columns = block_table.column_names
    result = pa.lib.table_to_blocks(options, block_table, categories, list(extension_columns.keys()))
    return [_reconstruct_block(item, columns, extension_columns) for item in result]