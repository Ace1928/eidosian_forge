import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.skip(reason='No longer supported')
def test_oracle_moves_missing_B(en_vocab):
    words = ['B', '52', 'Bomber']
    biluo_tags = [None, None, 'L-PRODUCT']
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {'words': words, 'entities': biluo_tags})
    moves = BiluoPushDown(en_vocab.strings)
    move_types = ('M', 'B', 'I', 'L', 'U', 'O')
    for tag in biluo_tags:
        if tag is None:
            continue
        elif tag == 'O':
            moves.add_action(move_types.index('O'), '')
        else:
            action, label = split_bilu_label(tag)
            moves.add_action(move_types.index('B'), label)
            moves.add_action(move_types.index('I'), label)
            moves.add_action(move_types.index('L'), label)
            moves.add_action(move_types.index('U'), label)
    moves.get_oracle_sequence(example)