import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import srsly
from jinja2 import Template
from thinc.api import Config
from wasabi import Printer, diff_strings
from .. import util
from ..language import DEFAULT_CONFIG_PRETRAIN_PATH
from ..schemas import RecommendationSchema
from ..util import SimpleFrozenList
from ._util import (

    Fill partial config file with default values. Will add all missing settings
    from the default config and will create all objects, check the registered
    functions for their default values and update the base config. This command
    can be used with a config generated via the training quickstart widget:
    https://spacy.io/usage/training#quickstart

    DOCS: https://spacy.io/api/cli#init-fill-config
    