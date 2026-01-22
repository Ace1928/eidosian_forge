import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def _sync_spell(self) -> None:
    """Sync this run with spell."""
    if not self._run:
        return
    try:
        env = os.environ
        self._interface.publish_config(key=('_wandb', 'spell_url'), val=env.get('SPELL_RUN_URL'))
        url = '{}/{}/{}/runs/{}'.format(self._api.app_url, self._run.entity, self._run.project, self._run.run_id)
        requests.put(env.get('SPELL_API_URL', 'https://api.spell.run') + '/wandb_url', json={'access_token': env.get('WANDB_ACCESS_TOKEN'), 'url': url}, timeout=2)
    except requests.RequestException:
        pass