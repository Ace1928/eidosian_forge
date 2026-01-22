import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..tokens import Doc, DocBin
from ..vocab import Vocab
from .augment import dont_augment
from .example import Example
class JsonlCorpus:
    """Iterate Example objects from a file or directory of jsonl
    formatted raw text files.

    path (Path): The directory or filename to read from.
    min_length (int): Minimum document length (in tokens). Shorter documents
        will be skipped. Defaults to 0, which indicates no limit.

    max_length (int): Maximum document length (in tokens). Longer documents will
        be skipped. Defaults to 0, which indicates no limit.
    limit (int): Limit corpus to a subset of examples, e.g. for debugging.
        Defaults to 0, which indicates no limit.

    DOCS: https://spacy.io/api/corpus#jsonlcorpus
    """
    file_type = 'jsonl'

    def __init__(self, path: Optional[Union[str, Path]], *, limit: int=0, min_length: int=0, max_length: int=0) -> None:
        self.path = util.ensure_path(path)
        self.min_length = min_length
        self.max_length = max_length
        self.limit = limit

    def __call__(self, nlp: 'Language') -> Iterator[Example]:
        """Yield examples from the data.

        nlp (Language): The current nlp object.
        YIELDS (Example): The example objects.

        DOCS: https://spacy.io/api/corpus#jsonlcorpus-call
        """
        for loc in walk_corpus(self.path, '.jsonl'):
            records = srsly.read_jsonl(loc)
            for record in records:
                doc = nlp.make_doc(record['text'])
                if self.min_length >= 1 and len(doc) < self.min_length:
                    continue
                elif self.max_length >= 1 and len(doc) >= self.max_length:
                    continue
                else:
                    words = [w.text for w in doc]
                    spaces = [bool(w.whitespace_) for w in doc]
                    yield Example(doc, Doc(nlp.vocab, words=words, spaces=spaces))