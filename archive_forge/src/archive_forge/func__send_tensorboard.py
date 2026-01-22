import atexit
import datetime
import fnmatch
import os
import queue
import sys
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import quote as url_quote
import wandb
from wandb.proto import wandb_internal_pb2  # type: ignore
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context, datastore, handler, sender, tb_watcher
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.util import check_and_warn_old
def _send_tensorboard(self, tb_root, tb_logdirs, send_manager):
    if self._entity is None:
        viewer, _ = send_manager._api.viewer_server_info()
        self._entity = viewer.get('entity')
    proto_run = wandb_internal_pb2.RunRecord()
    proto_run.run_id = self._run_id or wandb.util.generate_id()
    proto_run.project = self._project or wandb.util.auto_project_name(None)
    proto_run.entity = self._entity
    proto_run.telemetry.feature.sync_tfevents = True
    url = '{}/{}/{}/runs/{}'.format(self._app_url, url_quote(proto_run.entity), url_quote(proto_run.project), url_quote(proto_run.run_id))
    print('Syncing: %s ...' % url)
    sys.stdout.flush()
    record_q = queue.Queue()
    sender_record_q = queue.Queue()
    new_interface = InterfaceQueue(record_q)
    context_keeper = context.ContextKeeper()
    send_manager = sender.SendManager(settings=send_manager._settings, record_q=sender_record_q, result_q=queue.Queue(), interface=new_interface, context_keeper=context_keeper)
    record = send_manager._interface._make_record(run=proto_run)
    settings = wandb.Settings(root_dir=self._tmp_dir.name, run_id=proto_run.run_id, _start_datetime=datetime.datetime.now(), _start_time=time.time())
    settings_static = SettingsStatic(settings.to_proto())
    handle_manager = handler.HandleManager(settings=settings_static, record_q=record_q, result_q=None, stopped=False, writer_q=sender_record_q, interface=new_interface, context_keeper=context_keeper)
    filesystem.mkdir_exists_ok(settings.files_dir)
    send_manager.send_run(record, file_dir=settings.files_dir)
    watcher = tb_watcher.TBWatcher(settings, proto_run, new_interface, True)
    for tb in tb_logdirs:
        watcher.add(tb, True, tb_root)
        sys.stdout.flush()
    watcher.finish()
    progress_step = 0
    spinner_states = ['-', '\\', '|', '/']
    line = ' Uploading data to wandb\r'
    while len(handle_manager) > 0:
        data = next(handle_manager)
        handle_manager.handle(data)
        while len(send_manager) > 0:
            data = next(send_manager)
            send_manager.send(data)
        print_line = spinner_states[progress_step % 4] + line
        wandb.termlog(print_line, newline=False, prefix=True)
        progress_step += 1
    while len(send_manager) > 0:
        data = next(send_manager)
        send_manager.send(data)
    sys.stdout.flush()
    handle_manager.finish()
    send_manager.finish()