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
def test_serialize_subclassed_kb():
    """Check that IO of a custom KB works fine as part of an EL pipe."""
    config_string = '\n    [nlp]\n    lang = "en"\n    pipeline = ["entity_linker"]\n\n    [components]\n\n    [components.entity_linker]\n    factory = "entity_linker"\n    \n    [components.entity_linker.generate_empty_kb]\n    @misc = "kb_test.CustomEmptyKB.v1"\n    \n    [initialize]\n\n    [initialize.components]\n\n    [initialize.components.entity_linker]\n\n    [initialize.components.entity_linker.kb_loader]\n    @misc = "kb_test.CustomKB.v1"\n    entity_vector_length = 342\n    custom_field = 666\n    '

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

    @registry.misc('kb_test.CustomEmptyKB.v1')
    def empty_custom_kb() -> Callable[[Vocab, int], SubInMemoryLookupKB]:

        def empty_kb_factory(vocab: Vocab, entity_vector_length: int):
            return SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=0)
        return empty_kb_factory

    @registry.misc('kb_test.CustomKB.v1')
    def custom_kb(entity_vector_length: int, custom_field: int) -> Callable[[Vocab], SubInMemoryLookupKB]:

        def custom_kb_factory(vocab):
            kb = SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=custom_field)
            kb.add_entity('random_entity', 0.0, zeros(entity_vector_length))
            return kb
        return custom_kb_factory
    config = Config().from_str(config_string)
    nlp = load_model_from_config(config, auto_fill=True)
    nlp.initialize()
    entity_linker = nlp.get_pipe('entity_linker')
    assert type(entity_linker.kb) == SubInMemoryLookupKB
    assert entity_linker.kb.entity_vector_length == 342
    assert entity_linker.kb.custom_field == 666
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        entity_linker2 = nlp2.get_pipe('entity_linker')
        assert type(entity_linker2.kb) == SubInMemoryLookupKB
        assert entity_linker2.kb.entity_vector_length == 342
        assert entity_linker2.kb.custom_field == 666