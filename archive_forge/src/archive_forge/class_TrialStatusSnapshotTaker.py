from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
class TrialStatusSnapshotTaker(Callback):
    """Collects a sequence of statuses of trials as they progress.

    If all trials keep previous status, no snapshot is taken.
    """

    def __init__(self, snapshot: TrialStatusSnapshot):
        self._snapshot = snapshot

    def on_step_end(self, iteration, trials, **kwargs):
        new_snapshot = defaultdict(str)
        for trial in trials:
            new_snapshot[trial.trial_id] = trial.status
        self._snapshot.append(new_snapshot)