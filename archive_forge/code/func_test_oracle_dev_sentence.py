import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_oracle_dev_sentence(vocab, arc_eager):
    words_deps_heads = '\n        Rolls-Royce nn Inc.\n        Motor nn Inc.\n        Cars nn Inc.\n        Inc. nsubj said\n        said ROOT said\n        it nsubj expects\n        expects ccomp said\n        its poss sales\n        U.S. nn sales\n        sales nsubj steady\n        to aux steady\n        remain cop steady\n        steady xcomp expects\n        at prep steady\n        about quantmod 1,200\n        1,200 num cars\n        cars pobj at\n        in prep steady\n        1990 pobj in\n        . punct said\n    '
    expected_transitions = ['S', 'S', 'S', 'L-nn', 'L-nn', 'L-nn', 'S', 'L-nsubj', 'S', 'S', 'L-nsubj', 'R-ccomp', 'S', 'S', 'L-nn', 'L-poss', 'S', 'S', 'S', 'L-cop', 'L-aux', 'L-nsubj', 'R-xcomp', 'R-prep', 'S', 'L-quantmod', 'S', 'L-num', 'R-pobj', 'D', 'D', 'R-prep', 'R-pobj', 'D', 'D', 'D', 'D', 'R-punct', 'D', 'D']
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
    doc = Doc(Vocab(), words=gold_words)
    example = Example.from_dict(doc, {'heads': gold_heads, 'deps': gold_deps})
    ae_oracle_actions = arc_eager.get_oracle_sequence(example, _debug=False)
    ae_oracle_actions = [arc_eager.get_class_name(i) for i in ae_oracle_actions]
    assert ae_oracle_actions == expected_transitions