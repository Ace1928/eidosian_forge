import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
class TuneTerminalReporter(TuneReporterBase):

    def experiment_started(self, experiment_name: str, experiment_path: str, searcher_str: str, scheduler_str: str, total_num_samples: int, tensorboard_path: Optional[str]=None, **kwargs):
        if total_num_samples > sys.maxsize:
            total_num_samples_str = 'infinite'
        else:
            total_num_samples_str = str(total_num_samples)
        print(tabulate([['Search algorithm', searcher_str], ['Scheduler', scheduler_str], ['Number of trials', total_num_samples_str]], headers=['Configuration for experiment', experiment_name], tablefmt=AIR_TABULATE_TABLEFMT))
        super().experiment_started(experiment_name=experiment_name, experiment_path=experiment_path, searcher_str=searcher_str, scheduler_str=scheduler_str, total_num_samples=total_num_samples, tensorboard_path=tensorboard_path, **kwargs)

    def _print_heartbeat(self, trials, *sys_args, force: bool=False):
        if self._verbosity < self._heartbeat_threshold and (not force):
            return
        heartbeat_strs, table_data = self._get_heartbeat(trials, *sys_args, force_full_output=force)
        self._start_block('heartbeat')
        for s in heartbeat_strs:
            print(s)
        more_infos = []
        all_data = []
        fail_header = table_data.header
        for sub_table in table_data.data:
            all_data.extend(sub_table.trial_infos)
            if sub_table.more_info:
                more_infos.append(sub_table.more_info)
        print(tabulate(all_data, headers=fail_header, tablefmt=AIR_TABULATE_TABLEFMT, showindex=False))
        if more_infos:
            print(', '.join(more_infos))
        if not force:
            return
        trials_with_error = _get_trials_with_error(trials)
        if not trials_with_error:
            return
        self._start_block('status_errored')
        print(f'Number of errored trials: {len(trials_with_error)}')
        fail_header = ['Trial name', '# failures', 'error file']
        fail_table_data = [[str(trial), str(trial.run_metadata.num_failures) + ('' if trial.status == Trial.ERROR else '*'), trial.error_file] for trial in trials_with_error]
        print(tabulate(fail_table_data, headers=fail_header, tablefmt=AIR_TABULATE_TABLEFMT, showindex=False, colalign=('left', 'right', 'left')))
        if any((trial.status == Trial.TERMINATED for trial in trials_with_error)):
            print('* The trial terminated successfully after retrying.')