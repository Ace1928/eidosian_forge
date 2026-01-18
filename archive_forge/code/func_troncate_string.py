from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def troncate_string(s, max_length=25):
    return s if len(s) < max_length else s[:max_length] + '...'