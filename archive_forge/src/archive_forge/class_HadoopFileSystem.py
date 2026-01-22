import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
class HadoopFileSystem(ArrowFSWrapper):
    """A wrapper on top of the pyarrow.fs.HadoopFileSystem
    to connect it's interface with fsspec"""
    protocol = 'hdfs'

    def __init__(self, host='default', port=0, user=None, kerb_ticket=None, replication=3, extra_conf=None, **kwargs):
        """

        Parameters
        ----------
        host: str
            Hostname, IP or "default" to try to read from Hadoop config
        port: int
            Port to connect on, or default from Hadoop config if 0
        user: str or None
            If given, connect as this username
        kerb_ticket: str or None
            If given, use this ticket for authentication
        replication: int
            set replication factor of file for write operations. default value is 3.
        extra_conf: None or dict
            Passed on to HadoopFileSystem
        """
        from pyarrow.fs import HadoopFileSystem
        fs = HadoopFileSystem(host=host, port=port, user=user, kerb_ticket=kerb_ticket, replication=replication, extra_conf=extra_conf)
        super().__init__(fs=fs, **kwargs)

    @staticmethod
    def _get_kwargs_from_urls(path):
        ops = infer_storage_options(path)
        out = {}
        if ops.get('host', None):
            out['host'] = ops['host']
        if ops.get('username', None):
            out['user'] = ops['username']
        if ops.get('port', None):
            out['port'] = ops['port']
        if ops.get('url_query', None):
            queries = parse_qs(ops['url_query'])
            if queries.get('replication', None):
                out['replication'] = int(queries['replication'][0])
        return out