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
class BonusHandler(BaseHandler):

    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, worker_target):
        """
        Requests to /bonus/{worker_id} will give a bonus to that worker.

        Requires a reason, assignment_id, a unique token (for idempotence), and the
        bonus amount IN CENTS
        """
        data = tornado.escape.json_decode(self.request.body)
        reason = data['reason']
        assignment_id = data['assignment_id']
        bonus_cents = data['bonus_cents']
        token = data['bonus_token']
        dollar_amount = bonus_cents / 100.0
        self.mturk_manager.pay_bonus(worker_target, dollar_amount, assignment_id, reason, token)
        print('Bonused ${} to {} for reason {}'.format(dollar_amount, worker_target, reason))
        data = {'status': True}
        self.write(json.dumps(data))