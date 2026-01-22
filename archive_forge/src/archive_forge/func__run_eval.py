import json
import numpy as np
import os
import signal
from typing import Dict
from parlai.core.metrics import Metric
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.exceptions import StopTrainException
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import aggregate_named_reports, aggregate_unnamed_reports
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.worlds import create_task
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
from parlai.utils.distributed import (
from parlai.utils.misc import Timer, nice_report
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
def _run_eval(self, valid_worlds, opt, datatype, max_exs=-1, write_log=False):
    """
        Eval on validation/test data.

        :param valid_world:
            list of the pre-created validation worlds.
        :param opt:
            the options that specific the task, eval_task, etc
        :param datatype:
            the datatype to use, such as "valid" or "test"
        :param bool write_log:
            specifies to write metrics to file if the model_file is set
        :param int max_exs:
            limits the number of examples if max_exs > 0
        """
    logging.info(f'running eval: {datatype}')
    timer = Timer()
    reports = []
    max_exs_per_worker = max_exs / (len(valid_worlds) * num_workers())
    for v_world in valid_worlds:
        task_report = self._run_single_eval(opt, v_world, max_exs_per_worker)
        reports.append(task_report)
    tasks = [world.getID() for world in valid_worlds]
    named_reports = dict(zip(tasks, reports))
    report = aggregate_named_reports(named_reports, micro_average=self.opt.get('aggregate_micro', False))
    report = self._sync_metrics(report)
    metrics = f'{datatype}:\n{nice_report(report)}\n'
    logging.info(f'eval completed in {timer.time():.2f}s')
    logging.report(metrics)
    if write_log and opt.get('model_file') and is_primary_worker():
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(f'{metrics}\n')
        f.close()
    return report