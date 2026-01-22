import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _logging_permission_check(self):
    if self.is_test:
        return False
    if not os.path.exists(PARLAI_CRED_DIR):
        os.makedirs(PARLAI_CRED_DIR)
    if os.path.exists(PARLAI_MTURK_LOG_PERMISSION_FILE):
        with open(PARLAI_MTURK_LOG_PERMISSION_FILE, 'rb') as perm_file:
            permissions = pickle.load(perm_file)
            if permissions['allowed'] is True:
                return True
            elif time.time() - permissions['asked_time'] < TWO_WEEKS:
                return False
        os.remove(PARLAI_MTURK_LOG_PERMISSION_FILE)
    print("Would you like to help improve ParlAI-MTurk by providing some metrics? We would like to record acceptance, completion, and disconnect rates by worker. These metrics let us track the health of the platform. If you accept we'll collect this data on all of your future runs. We'd ask before collecting anything else, but currently we have no plans to. You can decline to snooze this request for 2 weeks.")
    selected = ''
    while selected not in ['y', 'Y', 'n', 'N']:
        selected = input('Share worker rates? (y/n): ')
        if selected not in ['y', 'Y', 'n', 'N']:
            print('Must type one of (Y/y/N/n)')
    if selected in ['y', 'Y']:
        print('Thanks for helping us make the platform better!')
    permissions = {'allowed': selected in ['y', 'Y'], 'asked_time': time.time()}
    with open(PARLAI_MTURK_LOG_PERMISSION_FILE, 'wb+') as perm_file:
        pickle.dump(permissions, perm_file)
        return permissions['allowed']