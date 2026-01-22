import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
import numpy as np
import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.utils import make_table
def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)