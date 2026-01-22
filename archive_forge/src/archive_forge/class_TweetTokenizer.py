import html
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
import regex
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
class TweetTokenizer:
    """
    Examples:

    ```python
    >>> # Tokenizer for tweets.
    >>> from nltk.tokenize import TweetTokenizer

    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    >>> # Examples using *strip_handles* and *reduce_len parameters*:
    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = "@remy: This is waaaaayyyy too much for you!!!!!!"
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    ```"""

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        Args:
            text: str

        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        """
        text = _replace_html_entities(text)
        if self.strip_handles:
            text = remove_handles(text)
        if self.reduce_len:
            text = reduce_lengthening(text)
        safe_text = HANG_RE.sub('\\1\\1\\1', text)
        words = WORD_RE.findall(safe_text)
        if not self.preserve_case:
            words = [x if EMOTICON_RE.search(x) else x.lower() for x in words]
        return words