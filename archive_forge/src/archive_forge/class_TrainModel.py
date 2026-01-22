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
@register_script('train_model', aliases=['tm', 'train'])
class TrainModel(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        self.train_loop = TrainLoop(self.opt)
        return self.train_loop.train()