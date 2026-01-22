import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional
import numpy
import typer
from tqdm import tqdm
from wasabi import msg
from .. import util
from ..language import Language
from ..tokens import Doc
from ..training import Corpus
from ._util import Arg, Opt, benchmark_cli, import_code, setup_gpu
Apply a statistic to repeated random samples of an array.