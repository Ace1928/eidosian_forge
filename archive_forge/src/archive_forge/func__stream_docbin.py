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
def _stream_docbin(path: Path, vocab: Vocab) -> Iterable[Doc]:
    """
    Stream Doc objects from DocBin.
    """
    docbin = DocBin().from_disk(path)
    for doc in docbin.get_docs(vocab):
        yield doc