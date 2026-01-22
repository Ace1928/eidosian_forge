import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_model_args(self):
    """
        Add arguments related to models such as model files.
        """
    model_args = self.add_argument_group('ParlAI Model Arguments')
    model_args.add_argument('-m', '--model', default=None, help='the model class name. can match parlai/agents/<model> for agents in that directory, or can provide a fully specified module for `from X import Y` via `-m X:Y` (e.g. `-m parlai.agents.seq2seq.seq2seq:Seq2SeqAgent`)')
    model_args.add_argument('-mf', '--model-file', default=None, help='model file name for loading and saving models')
    model_args.add_argument('-im', '--init-model', default=None, type=str, help='load model weights and dict from this file')
    model_args.add_argument('--dict-class', hidden=True, help='the class of the dictionary agent uses')