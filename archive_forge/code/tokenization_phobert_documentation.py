import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        