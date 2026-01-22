import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def _synset_from_pos_and_line(self, pos, data_file_line):
    synset = Synset(self)
    try:
        columns_str, gloss = data_file_line.strip().split('|')
        definition = re.sub('[\\"].*?[\\"]', '', gloss).strip()
        examples = re.findall('"([^"]*)"', gloss)
        for example in examples:
            synset._examples.append(example)
        synset._definition = definition.strip('; ')
        _iter = iter(columns_str.split())

        def _next_token():
            return next(_iter)
        synset._offset = int(_next_token())
        lexname_index = int(_next_token())
        synset._lexname = self._lexnames[lexname_index]
        synset._pos = _next_token()
        n_lemmas = int(_next_token(), 16)
        for _ in range(n_lemmas):
            lemma_name = _next_token()
            lex_id = int(_next_token(), 16)
            m = re.match('(.*?)(\\(.*\\))?$', lemma_name)
            lemma_name, syn_mark = m.groups()
            lemma = Lemma(self, synset, lemma_name, lexname_index, lex_id, syn_mark)
            synset._lemmas.append(lemma)
            synset._lemma_names.append(lemma._name)
        n_pointers = int(_next_token())
        for _ in range(n_pointers):
            symbol = _next_token()
            offset = int(_next_token())
            pos = _next_token()
            lemma_ids_str = _next_token()
            if lemma_ids_str == '0000':
                synset._pointers[symbol].add((pos, offset))
            else:
                source_index = int(lemma_ids_str[:2], 16) - 1
                target_index = int(lemma_ids_str[2:], 16) - 1
                source_lemma_name = synset._lemmas[source_index]._name
                lemma_pointers = synset._lemma_pointers
                tups = lemma_pointers[source_lemma_name, symbol]
                tups.append((pos, offset, target_index))
        try:
            frame_count = int(_next_token())
        except StopIteration:
            pass
        else:
            for _ in range(frame_count):
                plus = _next_token()
                assert plus == '+'
                frame_number = int(_next_token())
                frame_string_fmt = VERB_FRAME_STRINGS[frame_number]
                lemma_number = int(_next_token(), 16)
                if lemma_number == 0:
                    synset._frame_ids.append(frame_number)
                    for lemma in synset._lemmas:
                        lemma._frame_ids.append(frame_number)
                        lemma._frame_strings.append(frame_string_fmt % lemma._name)
                else:
                    lemma = synset._lemmas[lemma_number - 1]
                    lemma._frame_ids.append(frame_number)
                    lemma._frame_strings.append(frame_string_fmt % lemma._name)
    except ValueError as e:
        raise WordNetError(f'line {data_file_line!r}: {e}') from e
    for lemma in synset._lemmas:
        if synset._pos == ADJ_SAT:
            head_lemma = synset.similar_tos()[0]._lemmas[0]
            head_name = head_lemma._name
            head_id = '%02d' % head_lemma._lex_id
        else:
            head_name = head_id = ''
        tup = (lemma._name, WordNetCorpusReader._pos_numbers[synset._pos], lemma._lexname_index, lemma._lex_id, head_name, head_id)
        lemma._key = ('%s%%%d:%02d:%02d:%s:%s' % tup).lower()
    lemma_name = synset._lemmas[0]._name.lower()
    offsets = self._lemma_pos_offset_map[lemma_name][synset._pos]
    sense_index = offsets.index(synset._offset)
    tup = (lemma_name, synset._pos, sense_index + 1)
    synset._name = '%s.%s.%02i' % tup
    return synset