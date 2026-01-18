import re
import warnings
from typing import Iterator, List, Tuple
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.destructive import MacIntyreContractions
from nltk.tokenize.util import align_tokens
Duck-typing the abstract *tokenize()*.