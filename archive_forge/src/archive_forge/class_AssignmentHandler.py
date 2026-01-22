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
class AssignmentHandler(BaseHandler):

    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, assignment_target):
        assignments = [self.data_handler.get_assignment_data(assignment_target)]
        pairings = self.data_handler.get_pairings_for_assignment(assignment_target)
        processed_assignments = merge_assignments_with_pairings(assignments, pairings, 'assignment {}'.format(assignment_target))
        assignment = processed_assignments[0]
        run_id = assignment['run_id']
        onboarding_id = assignment['onboarding_id']
        conversation_id = assignment['conversation_id']
        worker_id = assignment['worker_id']
        onboard_data = None
        if onboarding_id is not None:
            onboard_data = MTurkDataHandler.get_conversation_data(run_id, onboarding_id, worker_id, self.state['is_sandbox'])
        assignment_content = {'onboarding': onboard_data, 'task': MTurkDataHandler.get_conversation_data(run_id, conversation_id, worker_id, self.state['is_sandbox']), 'task_name': '_'.join(run_id.split('_')[:-1])}
        task_name = '_'.join(run_id.split('_')[:-1])
        try:
            t = get_config_module(task_name)
            task_instructions = t.task_config['task_description']
        except ImportError:
            task_instructions = None
        data = {'assignment_details': assignment, 'assignment_content': assignment_content, 'assignment_instructions': task_instructions}
        self.write(json.dumps(data))