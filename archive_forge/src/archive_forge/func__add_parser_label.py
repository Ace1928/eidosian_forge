from typing import List
import pytest
import spacy
from spacy.training import Example
def _add_parser_label(parser, data):
    for _, annotations in data:
        for dep in annotations['deps']:
            parser.add_label(dep)