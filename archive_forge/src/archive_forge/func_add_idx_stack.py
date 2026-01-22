from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from extract_and_save_personas import main as main_extract
import numpy as np
import time
import os
import pickle
import random
def add_idx_stack(self):
    stack = [i for i in range(len(self.personas_name_list))]
    random.seed()
    random.shuffle(stack)
    self.idx_stack = stack + self.idx_stack