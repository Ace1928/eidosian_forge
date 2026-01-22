import argparse
from dataclasses import dataclass
from enum import Enum
import os.path
import tempfile
import typer
from typing import Optional
import requests
from ray.tune.experiment.config_parser import _make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
@dataclass
class CLIArguments:
    """Dataclass for CLI arguments and options. We use this class to keep track
    of common arguments, like "run" or "env" that would otherwise be duplicated."""
    Algo = typer.Option(None, '--algo', '--run', '-a', '-r', help=get_help('run'))
    AlgoRequired = typer.Option(..., '--algo', '--run', '-a', '-r', help=get_help('run'))
    Env = typer.Option(None, '--env', '-e', help=train_help.get('env'))
    EnvRequired = typer.Option(..., '--env', '-e', help=train_help.get('env'))
    Config = typer.Option('{}', '--config', '-c', help=get_help('config'))
    ConfigRequired = typer.Option(..., '--config', '-c', help=get_help('config'))
    ConfigFile = typer.Argument(..., help=train_help.get('config_file'))
    FileType = typer.Option(SupportedFileType.yaml, '--type', '-t', help=train_help.get('filetype'))
    Stop = typer.Option('{}', '--stop', '-s', help=get_help('stop'))
    ExperimentName = typer.Option('default', '--experiment-name', '-n', help=train_help.get('experiment_name'))
    V = typer.Option(False, '--log-info', '-v', help=train_help.get('v'))
    VV = typer.Option(False, '--log-debug', '-vv', help=train_help.get('vv'))
    Resume = typer.Option(False, help=train_help.get('resume'))
    NumSamples = typer.Option(1, help=get_help('num_samples'))
    CheckpointFreq = typer.Option(0, help=get_help('checkpoint_freq'))
    CheckpointAtEnd = typer.Option(True, help=get_help('checkpoint_at_end'))
    LocalDir = typer.Option(DEFAULT_RESULTS_DIR, help=train_help.get('local_dir'))
    Restore = typer.Option(None, help=get_help('restore'))
    Framework = typer.Option(None, help=train_help.get('framework'))
    ResourcesPerTrial = typer.Option(None, help=get_help('resources_per_trial'))
    KeepCheckpointsNum = typer.Option(None, help=get_help('keep_checkpoints_num'))
    CheckpointScoreAttr = typer.Option('training_iteration', help=get_help('sync_on_checkpoint'))
    UploadDir = typer.Option('', help=train_help.get('upload_dir'))
    Trace = typer.Option(False, help=train_help.get('trace'))
    LocalMode = typer.Option(False, help=train_help.get('local_mode'))
    Scheduler = typer.Option('FIFO', help=get_help('scheduler'))
    SchedulerConfig = typer.Option('{}', help=get_help('scheduler_config'))
    RayAddress = typer.Option(None, help=train_help.get('ray_address'))
    RayUi = typer.Option(False, help=train_help.get('ray_ui'))
    RayNumCpus = typer.Option(None, help=train_help.get('ray_num_cpus'))
    RayNumGpus = typer.Option(None, help=train_help.get('ray_num_gpus'))
    RayNumNodes = typer.Option(None, help=train_help.get('ray_num_nodes'))
    RayObjectStoreMemory = typer.Option(None, help=train_help.get('ray_object_store_memory'))
    WandBKey = typer.Option(None, '--wandb-key', help=train_help.get('wandb_key'))
    WandBProject = typer.Option(None, '--wandb-project', help=eval_help.get('wandb_project'))
    WandBRunName = typer.Option(None, '--wandb-run-name', help=eval_help.get('wandb_run_name'))
    Checkpoint = typer.Argument(None, help=eval_help.get('checkpoint'))
    Render = typer.Option(False, help=eval_help.get('render'))
    Steps = typer.Option(10000, help=eval_help.get('steps'))
    Episodes = typer.Option(0, help=eval_help.get('episodes'))
    Out = typer.Option(None, help=eval_help.get('out'))
    SaveInfo = typer.Option(False, help=eval_help.get('save_info'))
    UseShelve = typer.Option(False, help=eval_help.get('use_shelve'))
    TrackProgress = typer.Option(False, help=eval_help.get('track_progress'))