import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def type_no_sentperiod(self):
    """
        The type with its final period removed if it is marked as a
        sentence break.
        """
    if self.sentbreak:
        return self.type_no_period
    return self.type