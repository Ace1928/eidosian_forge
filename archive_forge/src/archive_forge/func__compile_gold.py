import math
import sys
from collections import Counter
from pathlib import Path
from typing import (
import numpy
import srsly
import typer
from wasabi import MESSAGES, Printer, msg
from .. import util
from ..compat import Literal
from ..language import Language
from ..morphology import Morphology
from ..pipeline import Morphologizer, SpanCategorizer, TrainablePipe
from ..pipeline._edit_tree_internals.edit_trees import EditTrees
from ..pipeline._parser_internals import nonproj
from ..pipeline._parser_internals.nonproj import DELIMITER
from ..schemas import ConfigSchemaTraining
from ..training import Example, remove_bilu_prefix
from ..training.initialize import get_sourced_components
from ..util import registry, resolve_dot_names
from ..vectors import Mode as VectorsMode
from ._util import (
def _compile_gold(examples: Sequence[Example], factory_names: List[str], nlp: Language, make_proj: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {'ner': Counter(), 'cats': Counter(), 'tags': Counter(), 'morphs': Counter(), 'deps': Counter(), 'words': Counter(), 'roots': Counter(), 'spancat': dict(), 'spans_length': dict(), 'spans_per_type': dict(), 'sb_per_type': dict(), 'ws_ents': 0, 'boundary_cross_ents': 0, 'n_words': 0, 'n_misaligned_words': 0, 'words_missing_vectors': Counter(), 'n_sents': 0, 'n_nonproj': 0, 'n_cycles': 0, 'n_cats_multilabel': 0, 'n_cats_bad_values': 0, 'texts': set(), 'lemmatizer_trees': set(), 'no_lemma_annotations': 0, 'partial_lemma_annotations': 0, 'n_low_cardinality_lemmas': 0}
    if 'trainable_lemmatizer' in factory_names:
        trees = EditTrees(nlp.vocab.strings)
    for eg in examples:
        gold = eg.reference
        doc = eg.predicted
        valid_words = [x.text for x in gold]
        data['words'].update(valid_words)
        data['n_words'] += len(valid_words)
        align = eg.alignment
        for token in doc:
            if token.orth_.isspace():
                continue
            if align.x2y.lengths[token.i] != 1:
                data['n_misaligned_words'] += 1
        data['texts'].add(doc.text)
        if len(nlp.vocab.vectors):
            for word in [t.text for t in doc]:
                if nlp.vocab.strings[word] not in nlp.vocab.vectors:
                    data['words_missing_vectors'].update([word])
        if 'ner' in factory_names:
            sent_starts = eg.get_aligned_sent_starts()
            for i, label in enumerate(eg.get_aligned_ner()):
                if label is None:
                    continue
                if label.startswith(('B-', 'U-', 'L-')) and doc[i].is_space:
                    data['ws_ents'] += 1
                if label.startswith(('B-', 'U-')):
                    combined_label = remove_bilu_prefix(label)
                    data['ner'][combined_label] += 1
                if sent_starts[i] and label.startswith(('I-', 'L-')):
                    data['boundary_cross_ents'] += 1
                elif label == '-':
                    data['ner']['-'] += 1
        if 'spancat' in factory_names or 'spancat_singlelabel' in factory_names:
            for spans_key in list(eg.reference.spans.keys()):
                if spans_key not in data['spancat']:
                    data['spancat'][spans_key] = Counter()
                for i, span in enumerate(eg.reference.spans[spans_key]):
                    if span.label_ is None:
                        continue
                    else:
                        data['spancat'][spans_key][span.label_] += 1
                if spans_key not in data['spans_length']:
                    data['spans_length'][spans_key] = dict()
                for span in gold.spans[spans_key]:
                    if span.label_ is None:
                        continue
                    if span.label_ not in data['spans_length'][spans_key]:
                        data['spans_length'][spans_key][span.label_] = []
                    data['spans_length'][spans_key][span.label_].append(len(span))
                if spans_key not in data['spans_per_type']:
                    data['spans_per_type'][spans_key] = dict()
                for span in gold.spans[spans_key]:
                    if span.label_ not in data['spans_per_type'][spans_key]:
                        data['spans_per_type'][spans_key][span.label_] = []
                    data['spans_per_type'][spans_key][span.label_].append(span)
                window_size = 1
                if spans_key not in data['sb_per_type']:
                    data['sb_per_type'][spans_key] = dict()
                for span in gold.spans[spans_key]:
                    if span.label_ not in data['sb_per_type'][spans_key]:
                        data['sb_per_type'][spans_key][span.label_] = {'start': [], 'end': []}
                    for offset in range(window_size):
                        sb_start_idx = span.start - (offset + 1)
                        if sb_start_idx >= 0:
                            data['sb_per_type'][spans_key][span.label_]['start'].append(gold[sb_start_idx:sb_start_idx + 1])
                        sb_end_idx = span.end + (offset + 1)
                        if sb_end_idx <= len(gold):
                            data['sb_per_type'][spans_key][span.label_]['end'].append(gold[sb_end_idx - 1:sb_end_idx])
        if 'textcat' in factory_names or 'textcat_multilabel' in factory_names:
            data['cats'].update(gold.cats)
            if any((val not in (0, 1) for val in gold.cats.values())):
                data['n_cats_bad_values'] += 1
            if list(gold.cats.values()).count(1) != 1:
                data['n_cats_multilabel'] += 1
        if 'tagger' in factory_names:
            tags = eg.get_aligned('TAG', as_string=True)
            data['tags'].update([x for x in tags if x is not None])
        if 'morphologizer' in factory_names:
            pos_tags = eg.get_aligned('POS', as_string=True)
            morphs = eg.get_aligned('MORPH', as_string=True)
            for pos, morph in zip(pos_tags, morphs):
                if pos is None or morph is None:
                    pass
                elif pos == '' and morph == '':
                    pass
                else:
                    label_dict = Morphology.feats_to_dict(morph)
                    if pos:
                        label_dict[Morphologizer.POS_FEAT] = pos
                    label = eg.reference.vocab.strings[eg.reference.vocab.morphology.add(label_dict)]
                    data['morphs'].update([label])
        if 'parser' in factory_names:
            aligned_heads, aligned_deps = eg.get_aligned_parse(projectivize=make_proj)
            data['deps'].update([x for x in aligned_deps if x is not None])
            for i, (dep, head) in enumerate(zip(aligned_deps, aligned_heads)):
                if head == i:
                    data['roots'].update([dep])
                    data['n_sents'] += 1
            if nonproj.is_nonproj_tree(aligned_heads):
                data['n_nonproj'] += 1
            if nonproj.contains_cycle(aligned_heads):
                data['n_cycles'] += 1
        if 'trainable_lemmatizer' in factory_names:
            if all((token.lemma == 0 for token in gold)):
                data['no_lemma_annotations'] += 1
                continue
            if any((token.lemma == 0 for token in gold)):
                data['partial_lemma_annotations'] += 1
            lemma_set = set()
            for token in gold:
                if token.lemma != 0:
                    lemma_set.add(token.lemma)
                    tree_id = trees.add(token.text, token.lemma_)
                    tree_str = trees.tree_to_str(tree_id)
                    data['lemmatizer_trees'].add(tree_str)
            if len(lemma_set) < 2 and len(gold) > 1:
                data['n_low_cardinality_lemmas'] += 1
    return data