import os
import re
import socket
from typing import Any, Optional
import wandb
import wandb.util
class TBXEventFileWriter(writer.EventFileWriter):

    def __init__(self, logdir: str, *args: Any, **kwargs: Any) -> None:
        logdir_hist.append(logdir)
        root_logdir_arg = root_logdir
        if len(set(logdir_hist)) > 1 and root_logdir == '':
            wandb.termwarn('When using several event log directories, please call `wandb.tensorboard.patch(root_logdir="...")` before `wandb.init`')
        hostname = socket.gethostname()
        search = re.search(f'-\\d+_{hostname}', logdir)
        if search:
            root_logdir_arg = logdir[:search.span()[1]]
        elif root_logdir is not None and (not os.path.abspath(logdir).startswith(os.path.abspath(root_logdir))):
            wandb.termwarn(f'Found log directory outside of given root_logdir, dropping given root_logdir for event file in {logdir}')
            root_logdir_arg = ''
        _notify_tensorboard_logdir(logdir, save=save, root_logdir=root_logdir_arg)
        super().__init__(logdir, *args, **kwargs)