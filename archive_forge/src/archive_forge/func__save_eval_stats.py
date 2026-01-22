from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import (
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import json
import random
from parlai.utils.distributed import (
def _save_eval_stats(opt, report):
    if not is_primary_worker:
        return
    report_fname = opt['report_filename']
    if report_fname == '':
        return
    if report_fname.startswith('.'):
        report_fname = opt['model_file'] + report_fname
    json_serializable_report = report
    for k, v in report.items():
        if isinstance(v, Metric):
            v = v.value()
        json_serializable_report[k] = v
    with open(report_fname, 'w') as f:
        logging.info(f'Saving model report to {report_fname}')
        json.dump({'opt': opt, 'report': json_serializable_report}, f, indent=4)
        f.write('\n')