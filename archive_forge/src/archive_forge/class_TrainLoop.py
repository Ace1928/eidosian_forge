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
class TrainLoop:
    """
    TrainLoop contains the core training loop logic.
    """

    def __init__(self, opt):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        trainstats_suffix = '.trainstats'
        if opt['load_from_checkpoint'] and opt.get('model_file') and os.path.isfile(opt['model_file'] + '.checkpoint'):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
            trainstats_suffix = '.checkpoint.trainstats'
        if not (opt.get('dict_file') or opt.get('model_file')):
            raise RuntimeError('WARNING: For train_model, please specify either a model_file or dict_file.')
        if 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            logging.info('building dictionary first...')
            build_dict(opt, skip_if_built=True)
        self.agent = create_agent(opt)
        self.agent.opt.log()
        self.world = create_task(opt, self.agent)
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        self.parleys = 0
        self.max_num_epochs = opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        self.max_train_time = opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        self.log_every_n_secs = opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        self.val_every_n_secs = opt['validation_every_n_secs'] if opt['validation_every_n_secs'] > 0 else float('inf')
        self.save_every_n_secs = opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        self.val_every_n_epochs = opt['validation_every_n_epochs'] if opt['validation_every_n_epochs'] > 0 else float('inf')
        if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
            opt['validation_metric_mode'] = 'min'
        elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
            opt['validation_metric_mode'] = 'max'
        if opt.get('validation_metric_mode') is None:
            opt['validation_metric_mode'] = 'max'
        self.last_valid_epoch = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.train_reports = []
        self.valid_reports = []
        self.best_valid = None
        self.impatience = 0
        self.saved = False
        self.valid_worlds = None
        self.opt = opt
        self._preempted_epochs = 0.0
        if opt.get('model_file') and os.path.isfile(opt['model_file'] + trainstats_suffix):
            with open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self.parleys = obj.get('parleys', 0)
                self._preempted_epochs = obj.get('total_epochs', 0)
                self.train_time.total = obj.get('train_time', 0)
                self.impatience = obj.get('impatience', 0)
                self.valid_reports = obj.get('valid_reports', [])
                self.train_reports = obj.get('train_reports', [])
                if 'best_valid' in obj:
                    self.best_valid = obj['best_valid']
                elif opt.get('model_file') and os.path.isfile(opt['model_file'] + '.best_valid'):
                    with open(opt['model_file'] + '.best_valid', 'r') as f:
                        x = f.readline()
                        self.best_valid = float(x)
                        f.close()
        if opt['tensorboard_log'] and is_primary_worker():
            self.tb_logger = TensorboardLogger(opt)

    def save_model(self, suffix=None):
        """
        Save the model to disk, possibly with a suffix.
        """
        if not is_primary_worker():
            return
        if not self.opt.get('model_file'):
            return
        fn = self.opt['model_file']
        if suffix:
            fn += suffix
        while True:
            try:
                self.agent.save(fn)
                self._save_train_stats(suffix)
                break
            except KeyboardInterrupt:
                pass

    def _safe_report(self, report: Dict[str, Metric]):
        return {k: v.value() if isinstance(v, Metric) else v for k, v in report.items()}

    def _save_train_stats(self, suffix=None):
        fn = self.opt['model_file']
        if suffix:
            fn += suffix
        fn += '.trainstats'
        with open(fn, 'w') as f:
            json.dump({'parleys': self.parleys, 'train_time': self.train_time.time(), 'total_epochs': self._total_epochs, 'train_reports': self.train_reports, 'valid_reports': self.valid_reports, 'best_valid': self.best_valid}, f, indent=4)

    def validate(self):
        """
        Perform a validation run, checking whether we should stop training.

        :return: boolean indicating whether training should stop
        :rtype: bool
        """
        opt = self.opt
        if self.valid_worlds is None:
            self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
        valid_report = self._run_eval(self.valid_worlds, opt, 'valid', opt['validation_max_exs'])
        v = self._safe_report(valid_report.copy())
        v['train_time'] = self.train_time.time()
        v['parleys'] = self.parleys
        v['total_exs'] = self._total_exs
        v['total_epochs'] = self._total_epochs
        self.valid_reports.append(v)
        if opt['tensorboard_log'] and is_primary_worker():
            valid_report['total_exs'] = self._total_exs
            self.tb_logger.log_metrics('valid', self.parleys, valid_report)
            self.tb_logger.flush()
        if opt.get('model_file') and opt.get('save_after_valid') and is_primary_worker():
            logging.info(f'saving model checkpoint: {opt['model_file']}.checkpoint')
            self.save_model('.checkpoint')
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)
        new_valid = valid_report[opt['validation_metric']]
        if isinstance(new_valid, Metric):
            new_valid = new_valid.value()
        if self.best_valid is None or self.valid_optim * new_valid > self.valid_optim * self.best_valid:
            logging.success('new best {}: {:.4g}{}'.format(opt['validation_metric'], new_valid, ' (previous best was {:.4g})'.format(self.best_valid) if self.best_valid is not None else ''))
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file') and is_primary_worker():
                logging.info(f'saving best valid model: {opt['model_file']}')
                self.save_model()
                self.saved = True
            if opt['validation_metric'] == 'accuracy' and self.best_valid >= opt['validation_cutoff']:
                logging.info('task solved! stopping.')
                return True
        else:
            self.impatience += 1
            logging.report('did not beat best {}: {} impatience: {}'.format(opt['validation_metric'], round(self.best_valid, 4), self.impatience))
        self.validate_time.reset()
        if opt['validation_patience'] > 0 and self.impatience >= opt['validation_patience']:
            logging.info('ran out of patience! stopping training.')
            return True
        return False

    def _run_single_eval(self, opt, valid_world, max_exs):
        valid_world.reset()
        cnt = 0
        max_cnt = max_exs if max_exs > 0 else float('inf')
        while not valid_world.epoch_done() and cnt < max_cnt:
            valid_world.parley()
            if cnt == 0 and opt['display_examples']:
                print(valid_world.display() + '\n~~')
                print(valid_world.report())
            cnt = valid_world.report().get('exs') or 0
        valid_report = valid_world.report()
        valid_world.reset()
        return valid_report

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

    def _sync_metrics(self, metrics):
        """
        Sync training metrics across workers.

        A handful of special cases are handled as exceptions, and the remaining metrics
        are simply averaged across workers.
        """
        if not is_distributed():
            return metrics
        all_versions = all_gather_list(metrics)
        return aggregate_unnamed_reports(all_versions)

    def _compute_eta(self, epochs_completed, time_elapsed):
        """
        Compute the estimated seconds remaining in training.

        :param float epochs_completed: number of epochs already completed.
        :param float time_elapsed: total time spent already, in seconds.
        :return: ETA in seconds, or None if not computable
        """
        eta = None
        max_epochs = self.opt.get('num_epochs', 0)
        if max_epochs > 0 and epochs_completed > 0:
            epoch_progress = epochs_completed / max_epochs
            eta = (1 - epoch_progress) * time_elapsed / epoch_progress
        max_training_time = self.opt.get('max_training_time', -1)
        if max_training_time > 0:
            time_left = max_training_time - time_elapsed
            if eta is None or time_left < eta:
                eta = time_left
        return eta

    def log(self):
        """
        Output a training log entry.
        """
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        train_report = self.world.report()
        train_report = self._sync_metrics(train_report)
        self.world.reset_metrics()
        train_report_trainstats = self._safe_report(train_report)
        train_report_trainstats['total_epochs'] = self._total_epochs
        train_report_trainstats['total_exs'] = self._total_exs
        train_report_trainstats['parleys'] = self.parleys
        train_report_trainstats['train_time'] = self.train_time.time()
        self.train_reports.append(train_report_trainstats)
        logs.append(f'time:{self.train_time.time():.0f}s')
        logs.append(f'total_exs:{self._total_exs}')
        if self._total_epochs >= 0:
            logs.append(f'epochs:{self._total_epochs:.2f}')
        time_left = self._compute_eta(self._total_epochs, self.train_time.time())
        if time_left is not None:
            logs.append(f'time_left:{max(0, time_left):.0f}s')
        log = '{}\n{}\n'.format(' '.join(logs), nice_report(train_report))
        logging.info(log)
        self.log_time.reset()
        if opt['tensorboard_log'] and is_primary_worker():
            self.tb_logger.log_metrics('train', self.parleys, train_report)

    def train(self):
        """
        Perform a training run.

        :return: tuple of reports (validation_report, test_report)
        """
        logging.info('training...')
        opt = self.opt
        world = self.world
        with world:
            while True:
                try:
                    world.parley()
                except StopTrainException:
                    if is_distributed():
                        raise RuntimeError('StopTrainException not supported for distributed mode')
                    break
                self.parleys += 1
                self._total_epochs = self._preempted_epochs + sum(all_gather_list(world.get_total_epochs()))
                exs_per_epoch = world.num_examples()
                self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))
                train_time, log_time, validate_time = sync_object((self.train_time.time(), self.log_time.time(), self.validate_time.time()))
                if self._total_epochs >= self.max_num_epochs:
                    self.log()
                    logging.info(f'num_epochs completed:{self.max_num_epochs} time elapsed:{train_time}s')
                    break
                if train_time > self.max_train_time:
                    logging.info(f'max_train_time elapsed:{train_time}s')
                    break
                if log_time > self.log_every_n_secs:
                    self.log()
                if validate_time > self.val_every_n_secs or self._total_epochs - self.last_valid_epoch >= self.val_every_n_epochs:
                    try:
                        self.log()
                        world.reset_metrics()
                        stop_training = self.validate()
                    except StopTrainException:
                        if is_distributed():
                            raise RuntimeError('StopTrainException not supported for distributed mode')
                        break
                    self.log_time.reset()
                    self.last_valid_epoch = self._total_epochs
                    if stop_training:
                        break
                    world.reset_metrics()
                if self.save_time.time() > self.save_every_n_secs and opt.get('model_file') and is_primary_worker():
                    logging.info(f'saving model checkpoint: {opt['model_file']}.checkpoint')
                    if opt['tensorboard_log'] and is_primary_worker():
                        self.tb_logger.flush()
                    self.save_model('.checkpoint')
                    self.save_time.reset()
        if not self.saved and is_primary_worker():
            self.save_model()
        sync_object(None)
        if opt.get('model_file'):
            del world
            del self.world
            del self.agent
            del self.valid_worlds
            self.agent = create_agent(opt)
        valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
        max_exs = opt['validation_max_exs'] if opt.get('short_final_eval') else -1
        v_report = self._run_eval(valid_worlds, opt, 'valid', max_exs, write_log=True)
        test_worlds = load_eval_worlds(self.agent, opt, 'test')
        t_report = self._run_eval(test_worlds, opt, 'test', max_exs, write_log=True)
        if valid_worlds:
            for valid_world in valid_worlds:
                valid_world.shutdown()
        if test_worlds:
            for test_world in test_worlds:
                test_world.shutdown()
        print_announcements(opt)
        return (v_report, t_report)