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
def _signf_level(p):
    if p < 0.001:
        return ('***', 'p<.001')
    elif p < 0.01:
        return ('**', 'p<.01')
    elif p < 0.05:
        return ('*', 'p<.05')
    else:
        return ('', 'p>.05')