import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Type
from torch.utils.data import DataLoader, Dataset, IterableDataset
import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.util import PublicAPI
@PublicAPI(stability='beta')
class RayTrainReportCallback(TrainerCallback):
    """A simple callback to report checkpoints and metrics to Ray Tarin.

    This callback is a subclass of `transformers.TrainerCallback
    <https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback>`_
    and overrides the `TrainerCallback.on_save()` method. After
    a new checkpoint get saved, it fetches the latest metric dictionary
    from `TrainerState.log_history` and reports it with the latest checkpoint
    to Ray Train.

    Checkpoints will be saved in the following structure::

        checkpoint_00000*/   Ray Train Checkpoint
        └─ checkpoint/       Hugging Face Transformers Checkpoint

    For customized reporting and checkpointing logic, implement your own
    `transformers.TrainerCallback` following this user
    guide: :ref:`Saving and Loading Checkpoints <train-dl-saving-checkpoints>`.

    Note that users should ensure that the logging, evaluation, and saving frequencies
    are properly configured so that the monitoring metric is always up-to-date
    when `transformers.Trainer` saves a checkpoint.

    Suppose the monitoring metric is reported from evaluation stage:

    Some valid configurations:
        - evaluation_strategy == save_strategy == "epoch"
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps == 0

    Some invalid configurations:
        - evaluation_strategy != save_strategy
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps != 0

    """
    CHECKPOINT_NAME = 'checkpoint'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_RAYTRAINREPORTCALLBACK, '1')

    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""
        with TemporaryDirectory() as tmpdir:
            metrics = {}
            for log in state.log_history:
                metrics.update(log)
            source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)
            target_ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_NAME)
            shutil.copytree(source_ckpt_path, target_ckpt_path)
            checkpoint = Checkpoint.from_directory(tmpdir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)