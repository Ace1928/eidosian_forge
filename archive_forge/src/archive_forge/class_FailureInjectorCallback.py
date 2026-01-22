from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
class FailureInjectorCallback(Callback):
    """Adds random failure injection to the TrialExecutor."""

    def __init__(self, config_path='~/ray_bootstrap_config.yaml', probability=0.1, time_between_checks=0, disable=False):
        self.probability = probability
        self.config_path = Path(config_path).expanduser().as_posix()
        self.disable = disable
        self.time_between_checks = time_between_checks
        self.last_fail_check = time.monotonic()

    def on_step_begin(self, **info):
        if not os.path.exists(self.config_path):
            return
        if time.monotonic() < self.last_fail_check + self.time_between_checks:
            return
        self.last_fail_check = time.monotonic()
        import click
        from ray.autoscaler._private.commands import kill_node
        failures = 0
        max_failures = 3
        if random.random() < self.probability and (not self.disable):
            should_terminate = random.random() < self.probability
            while failures < max_failures:
                try:
                    kill_node(self.config_path, yes=True, hard=should_terminate, override_cluster_name=None)
                    return
                except click.exceptions.ClickException:
                    failures += 1
                    logger.exception('Killing random node failed in attempt {}. Retrying {} more times'.format(str(failures), str(max_failures - failures)))