import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_oracle_bad_tokenization(vocab, arc_eager):
    words_deps_heads = '\n        [catalase] dep is\n        : punct is\n        that nsubj is\n        is root is\n        bad comp is\n    '
    gold_words = []
    gold_deps = []
    gold_heads = []
    for line in words_deps_heads.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        word, dep, head = line.split()
        gold_words.append(word)
        gold_deps.append(dep)
        gold_heads.append(head)
    gold_heads = [gold_words.index(head) for head in gold_heads]
    for dep in gold_deps:
        arc_eager.add_action(2, dep)
        arc_eager.add_action(3, dep)
    reference = Doc(Vocab(), words=gold_words, deps=gold_deps, heads=gold_heads)
    predicted = Doc(reference.vocab, words=['[', 'catalase', ']', ':', 'that', 'is', 'bad'])
    example = Example(predicted=predicted, reference=reference)
    ae_oracle_actions = arc_eager.get_oracle_sequence(example, _debug=False)
    ae_oracle_actions = [arc_eager.get_class_name(i) for i in ae_oracle_actions]
    assert ae_oracle_actions