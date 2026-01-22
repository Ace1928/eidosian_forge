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
class TaskRunHandler(BaseHandler):

    def initialize(self, app):
        self.app = app

    def post(self, task_target):
        """
        Requests to /run_task/{task_id} will launch a task locally for the given task.

        It will die after 20 mins if it doesn't end on its own.
        """
        try:
            t = get_run_module(task_target)
            conf = get_config_module(task_target)
            t.MTurkManager = MockTurkManager
            MockTurkManager.current_manager = None
            task_thread = threading.Thread(target=t.main, name='Demo-Thread')
            task_thread.start()
            while MockTurkManager.current_manager is None:
                time.sleep(1)
            time.sleep(1)
            manager = MockTurkManager.current_manager
            self.app.task_manager = manager
            for agent in manager.agents:
                manager.worker_alive(agent.worker_id, agent.hit_id, agent.assignment_id)
            data = {'started': True, 'data': [agent.get_update_packet() for agent in manager.agents], 'task_config': conf.task_config}
            self.write(json.dumps(data))
        except Exception as e:
            data = {'error': e}
            print(repr(e))
            print(traceback.format_exc())
            self.write(json.dumps(data))