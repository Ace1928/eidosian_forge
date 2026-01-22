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
def _run_socket(self):
    time.sleep(2)
    asyncio.set_event_loop(asyncio.new_event_loop())
    while self.alive and self.app.task_manager is not None:
        try:
            self.write_message(json.dumps({'data': [agent.get_update_packet() for agent in self.app.task_manager.agents], 'command': 'sync'}))
            time.sleep(0.2)
        except tornado.websocket.WebSocketClosedError:
            self.alive = False
            self.app.task_manager.timeout_all_agents()