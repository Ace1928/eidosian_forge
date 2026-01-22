from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir
class SubInMemoryLookupKB(InMemoryLookupKB):

    def __init__(self, vocab, entity_vector_length, custom_field):
        super().__init__(vocab, entity_vector_length)
        self.custom_field = custom_field

    def to_disk(self, path, exclude: Iterable[str]=SimpleFrozenList()):
        """We overwrite InMemoryLookupKB.to_disk() to ensure that self.custom_field is stored as well."""
        path = ensure_path(path)
        if not path.exists():
            path.mkdir(parents=True)
        if not path.is_dir():
            raise ValueError(Errors.E928.format(loc=path))

        def serialize_custom_fields(file_path: Path) -> None:
            srsly.write_json(file_path, {'custom_field': self.custom_field})
        serialize = {'contents': lambda p: self.write_contents(p), 'strings.json': lambda p: self.vocab.strings.to_disk(p), 'custom_fields': lambda p: serialize_custom_fields(p)}
        util.to_disk(path, serialize, exclude)

    def from_disk(self, path, exclude: Iterable[str]=SimpleFrozenList()):
        """We overwrite InMemoryLookupKB.from_disk() to ensure that self.custom_field is loaded as well."""
        path = ensure_path(path)
        if not path.exists():
            raise ValueError(Errors.E929.format(loc=path))
        if not path.is_dir():
            raise ValueError(Errors.E928.format(loc=path))

        def deserialize_custom_fields(file_path: Path) -> None:
            self.custom_field = srsly.read_json(file_path)['custom_field']
        deserialize: Dict[str, Callable[[Any], Any]] = {'contents': lambda p: self.read_contents(p), 'strings.json': lambda p: self.vocab.strings.from_disk(p), 'custom_fields': lambda p: deserialize_custom_fields(p)}
        util.from_disk(path, deserialize, exclude)