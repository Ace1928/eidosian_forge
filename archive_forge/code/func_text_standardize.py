import json
import os
import re
import unicodedata
from typing import Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('(-+|~+|!+|"+|;+|\\?+|\\++|,+|\\)+|\\(+|\\\\+|\\/+|\\*+|\\[+|\\]+|}+|{+|\\|+|_+)', ' \\1 ', text)
    text = re.sub('\\s*\\n\\s*', ' \n ', text)
    text = re.sub('[^\\S\\n]+', ' ', text)
    return text.strip()