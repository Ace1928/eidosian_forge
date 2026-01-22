from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.core.worlds import validate
from parlai.mturk.tasks.personachat.personachat_chat.extract_and_save_personas import (
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle
import random
def add_done_persona(self, idx):
    self.done_personas.append(idx)
    num_done = len(self.done_personas)
    print('Number of completed personas:', num_done)
    print('Completed personas:', sorted(self.done_personas))