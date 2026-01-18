import logging
from pathlib import Path
from typing import Optional
import srsly
import typer
from wasabi import msg
from .. import util
from ..language import Language
from ..training.initialize import convert_vectors, init_nlp
from ._util import (
Generate JSON files for the labels in the data. This helps speed up the
    training process, since spaCy won't have to preprocess the data to
    extract the labels.