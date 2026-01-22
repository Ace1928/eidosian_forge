import json
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, List, Optional
import requests
from ..utils import logging
from . import BaseTransformersCLICommand
class LfsUploadCommand:

    def __init__(self, args):
        self.args = args

    def run(self):
        init_msg = json.loads(sys.stdin.readline().strip())
        if not (init_msg.get('event') == 'init' and init_msg.get('operation') == 'upload'):
            write_msg({'error': {'code': 32, 'message': 'Wrong lfs init operation'}})
            sys.exit(1)
        write_msg({})
        while True:
            msg = read_msg()
            if msg is None:
                sys.exit(0)
            oid = msg['oid']
            filepath = msg['path']
            completion_url = msg['action']['href']
            header = msg['action']['header']
            chunk_size = int(header.pop('chunk_size'))
            presigned_urls: List[str] = list(header.values())
            parts = []
            for i, presigned_url in enumerate(presigned_urls):
                with FileSlice(filepath, seek_from=i * chunk_size, read_limit=chunk_size) as data:
                    r = requests.put(presigned_url, data=data)
                    r.raise_for_status()
                    parts.append({'etag': r.headers.get('etag'), 'partNumber': i + 1})
                    write_msg({'event': 'progress', 'oid': oid, 'bytesSoFar': (i + 1) * chunk_size, 'bytesSinceLast': chunk_size})
            r = requests.post(completion_url, json={'oid': oid, 'parts': parts})
            r.raise_for_status()
            write_msg({'event': 'complete', 'oid': oid})