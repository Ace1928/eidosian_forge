import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import binom_test
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML
def _extract_response_data(self, data: Dict[str, Any], task_data: Dict[str, Any], hit: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Extract response data from task data.

        :param data:
            full data from a given turker
        :param task_data:
            data from one "task" completion (i.e. one dialogue comparison)
        :param hit:
            hit data
        :param response_data:
            turker's corresponding response data corresponding to the task

        :return response:
            Turker's response data from the task
        """
    response: Dict[str, Any] = {'run_id': self.run_id, 'worker': data['worker_id'], 'time_taken': hit['task_end'] - hit['task_start'], 'question': data['task_data'][0]['task_specs']['question'], 'conversation_id': hit['conversation_id']}
    onboarding = task_data['task_specs'].get('is_onboarding', False)
    choice = response_data['speakerChoice']
    if onboarding:
        response['correct'] = choice == task_data['pairing_dict']['correct_answer']
    else:
        response['correct'] = -1
    speakers_to_eval = sorted(task_data['pairing_dict']['speakers_to_eval'])
    response.update({'winner': choice, 'loser': speakers_to_eval[1 - speakers_to_eval.index(choice)], 'eval_choice_0': speakers_to_eval[0], 'eval_choice_1': speakers_to_eval[1], 'reason': response_data['textReason'], 'is_onboarding': onboarding, 'matchup': f'{'__vs__'.join(speakers_to_eval)}', 'pairing_id': task_data['pair_id'], 'dialogue_lengths': {task_data['task_specs']['model_left']['name']: len(task_data['task_specs']['model_left']['dialogue']), task_data['task_specs']['model_right']['name']: len(task_data['task_specs']['model_right']['dialogue'])}, 'speaker_model_mapping': [task_data['task_specs']['model_left']['name'], task_data['task_specs']['model_right']['name']]})
    return response