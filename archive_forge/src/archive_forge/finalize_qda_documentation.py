import json
from glob import glob
from os.path import join as pjoin
from parlai.core.params import ParlaiParser

Adapted from https://github.com/facebookresearch/ELI5/blob/master/data_creation/finalize_qda.py
to use data directory rather than a hard-coded directory
