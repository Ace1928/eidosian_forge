import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def trove_classifier(value: str) -> bool:
    return value in _trove_classifiers or value.lower().startswith('private ::')