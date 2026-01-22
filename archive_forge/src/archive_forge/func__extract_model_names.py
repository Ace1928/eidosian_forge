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
def _extract_model_names(self):
    """
        Extract the model nicknames from the dataframe.
        """
    df = self.dataframe
    df = df[df['run_id'] == self.run_id]
    matchups = list(df.matchup.unique())
    models = set()
    combos = set()
    for matchup in matchups:
        model1, model2 = matchup.split('__vs__')
        models.add(model1)
        models.add(model2)
        combos.add(tuple(sorted((model1, model2))))
    self.models = list(models)
    self.models.sort()
    self.combos = list(combos)
    self.combos.sort()