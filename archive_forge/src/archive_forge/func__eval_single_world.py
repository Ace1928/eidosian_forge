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
def _eval_single_world(opt, agent, task):
    logging.info(f'Evaluating task {task} using datatype {opt.get('datatype')}.')
    world_logger = WorldLogger(opt) if opt['save_world_logs'] else None
    task_opt = opt.copy()
    task_opt['task'] = task
    world = create_task(task_opt, agent)
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0
    total_cnt = world.num_examples()
    if is_distributed():
        logging.warn('Progress bar is approximate in distributed mode.')
    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        if world_logger is not None:
            world_logger.log(world)
        if opt['display_examples']:
            print(world.display() + '\n~~')
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report.get('exs', 0), min(max_cnt, total_cnt), report)
            logging.info(text)
    report = aggregate_unnamed_reports(all_gather_list(world.report()))
    world.reset()
    if world_logger is not None:
        world_logger.reset()
        base_outfile = opt['report_filename'].split('.')[0]
        if is_distributed():
            rank = get_rank()
            outfile = base_outfile + f'_{task}_{rank}_replies.jsonl'
        else:
            outfile = base_outfile + f'_{task}_replies.jsonl'
        world_logger.write(outfile, world, file_format=opt['save_format'])
    return report