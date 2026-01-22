import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class MySQLSink(IOBase):
    """
    Very simple frontend for storing values into MySQL database.

    Examples
    --------

    >>> sql = MySQLSink(input_names=['subject_id', 'some_measurement'])
    >>> sql.inputs.database_name = 'my_database'
    >>> sql.inputs.table_name = 'experiment_results'
    >>> sql.inputs.username = 'root'
    >>> sql.inputs.password = 'secret'
    >>> sql.inputs.subject_id = 's1'
    >>> sql.inputs.some_measurement = 11.4
    >>> sql.run() # doctest: +SKIP

    """
    input_spec = MySQLSinkInputSpec

    def __init__(self, input_names, **inputs):
        super(MySQLSink, self).__init__(**inputs)
        self._input_names = ensure_list(input_names)
        add_traits(self.inputs, [name for name in self._input_names])

    def _list_outputs(self):
        """Execute this module."""
        import MySQLdb
        if isdefined(self.inputs.config):
            conn = MySQLdb.connect(db=self.inputs.database_name, read_default_file=self.inputs.config)
        else:
            conn = MySQLdb.connect(host=self.inputs.host, user=self.inputs.username, passwd=self.inputs.password, db=self.inputs.database_name)
        c = conn.cursor()
        c.execute('REPLACE INTO %s (' % self.inputs.table_name + ','.join(self._input_names) + ') VALUES (' + ','.join(['%s'] * len(self._input_names)) + ')', [getattr(self.inputs, name) for name in self._input_names])
        conn.commit()
        c.close()
        return None