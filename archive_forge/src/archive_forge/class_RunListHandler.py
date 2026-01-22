from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import importlib
import inspect
import json
import logging
import os
import time
import threading
import traceback
import asyncio
import sh
import shlex
import shutil
import subprocess
import uuid
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.webapp.run_mocks.mock_turk_manager import MockTurkManager
from typing import Dict, Any
from parlai import __path__ as parlai_path  # type: ignore
class RunListHandler(BaseHandler):

    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def post(self):
        req = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))
        self.write(json.dumps({'t': 'testing!', 'req': req}))

    def get(self):
        results = self.data_handler.get_all_run_data()
        processed_results = []
        for res in results:
            processed_results.append(row_to_dict(res))
        for result in processed_results:
            result['run_status'] = 'unimplemented'
        self.write(json.dumps(processed_results))