from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast
import srsly
import tqdm
from wasabi import msg
from ..tokens import Doc, DocBin
from ..util import ensure_path, load_model
from ..vocab import Vocab
from ._util import Arg, Opt, app, import_code, setup_gpu, walk_directory
def _stream_jsonl(path: Path, field: str) -> Iterable[str]:
    """
    Stream "text" field from JSONL. If the field "text" is
    not found it raises error.
    """
    for entry in srsly.read_jsonl(path):
        if field not in entry:
            msg.fail(f"{path} does not contain the required '{field}' field.", exits=1)
        else:
            yield entry[field]