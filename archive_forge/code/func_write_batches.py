from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
def write_batches(batches, as_table=False):
    with format_fixture._get_writer(pa.MockOutputStream(), schema) as writer:
        if as_table:
            table = pa.Table.from_batches(batches)
            writer.write_table(table)
        else:
            for batch in batches:
                writer.write_batch(batch)
        return writer.stats