from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def _load_conversation_data(self):
    """
        Load conversation data.

        Loads in the data from the pairs filepath.
        """
    pairs_path = self.opt.get('pairings_filepath')
    if not os.path.exists(pairs_path):
        raise RuntimeError('You MUST specify a valid pairings filepath')
    with open(pairs_path) as pf:
        for i, l in enumerate(pf.readlines()):
            convo_pair = json.loads(l.strip())
            eval_speakers = [s for d in convo_pair['dialogue_dicts'] for s in d['speakers'] if s in convo_pair['speakers_to_eval']]
            assert eval_speakers == convo_pair['speakers_to_eval']
            model_left_idx = random.choice([0, 1])
            task = {'task_specs': {'s1_choice': self.opt['s1_choice'], 's2_choice': self.opt['s2_choice'], 'question': self.opt['question'], 'is_onboarding': convo_pair['is_onboarding'], 'model_left': {'name': eval_speakers[model_left_idx], 'dialogue': convo_pair['dialogue_dicts'][model_left_idx]['dialogue']}, 'model_right': {'name': eval_speakers[1 - model_left_idx], 'dialogue': convo_pair['dialogue_dicts'][1 - model_left_idx]['dialogue']}}, 'pairing_dict': convo_pair, 'pair_id': i}
            if convo_pair.get('is_onboarding'):
                self.onboarding_tasks.append(task)
            else:
                self.desired_tasks.append(task)