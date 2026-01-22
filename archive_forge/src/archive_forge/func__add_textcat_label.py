from typing import List
import pytest
import spacy
from spacy.training import Example
def _add_textcat_label(textcat, data):
    for _, annotations in data:
        for cat in annotations['cats']:
            textcat.add_label(cat)