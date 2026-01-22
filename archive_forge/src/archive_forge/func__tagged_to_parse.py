import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def _tagged_to_parse(self, tagged_tokens):
    """
        Convert a list of tagged tokens to a chunk-parse tree.
        """
    sent = Tree('S', [])
    for tok, tag in tagged_tokens:
        if tag == 'O':
            sent.append(tok)
        elif tag.startswith('B-'):
            sent.append(Tree(tag[2:], [tok]))
        elif tag.startswith('I-'):
            if sent and isinstance(sent[-1], Tree) and (sent[-1].label() == tag[2:]):
                sent[-1].append(tok)
            else:
                sent.append(Tree(tag[2:], [tok]))
    return sent