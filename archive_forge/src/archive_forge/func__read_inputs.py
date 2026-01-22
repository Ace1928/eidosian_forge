import cProfile
import itertools
import pstats
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union
import srsly
import tqdm
import typer
from wasabi import Printer, msg
from ..language import Language
from ..util import load_model
from ._util import NAME, Arg, Opt, app, debug_cli
def _read_inputs(loc: Union[Path, str], msg: Printer) -> Iterator[str]:
    if loc == '-':
        msg.info('Reading input from sys.stdin')
        file_ = sys.stdin
        file_ = (line.encode('utf8') for line in file_)
    else:
        input_path = Path(loc)
        if not input_path.exists() or not input_path.is_file():
            msg.fail('Not a valid input data file', loc, exits=1)
        msg.info(f'Using data from {input_path.parts[-1]}')
        file_ = input_path.open()
    for line in file_:
        data = srsly.json_loads(line)
        text = data['text']
        yield text