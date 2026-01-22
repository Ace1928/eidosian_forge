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
class Quartiles:
    """Calculate the q1, q2, q3 quartiles and the inter-quartile range (iqr)
    of a sample."""
    q1: float
    q2: float
    q3: float
    iqr: float

    def __init__(self, sample: numpy.ndarray) -> None:
        self.q1 = numpy.quantile(sample, 0.25)
        self.q2 = numpy.quantile(sample, 0.5)
        self.q3 = numpy.quantile(sample, 0.75)
        self.iqr = self.q3 - self.q1