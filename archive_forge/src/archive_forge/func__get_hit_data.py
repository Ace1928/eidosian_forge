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
def _get_hit_data(self, hit: Dict[str, Any], logger: MTurkDataHandler) -> Optional[Dict[str, Any]]:
    """
        Return data for a given hit.

        If the HIT is corrupt for whatever reason, we return None

        :param hit:
            HIT information dict
        :param logger:
            Data handler

        :return data:
            Optional dict with the hit data
        """
    try:
        full_data: Dict[str, Any] = logger.get_full_conversation_data(self.run_id, hit['conversation_id'], self.is_sandbox)
    except FileNotFoundError:
        print(f'WARNING: Data for run_id `{self.run_id}` not found for conversation id {hit['conversation_id']}')
        return None
    data: Dict[str, Any] = next(iter(full_data['worker_data'].values()))
    if not ('task_data' in data['response'] and len(data['response']['task_data']) > 0):
        return None
    elif len(data['response']['task_data']) != len(data['task_data']):
        raise ValueError('Saved task data does not match response task data')
    return data