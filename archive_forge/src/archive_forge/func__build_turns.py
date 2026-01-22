import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
def _build_turns(self, episode):
    turns = []
    for act_pair in episode['dialog']:
        for act in act_pair:
            turns.append(Turn(**act))
    return turns