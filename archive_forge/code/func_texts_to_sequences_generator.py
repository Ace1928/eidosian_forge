import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def texts_to_sequences_generator(self, texts):
    """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Args:
            texts: A list of texts (strings).

        Yields:
            Yields individual sequences.
        """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for text in texts:
        if self.char_level or isinstance(text, list):
            if self.lower:
                if isinstance(text, list):
                    text = [text_elem.lower() for text_elem in text]
                else:
                    text = text.lower()
            seq = text
        elif self.analyzer is None:
            seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)
        else:
            seq = self.analyzer(text)
        vect = []
        for w in seq:
            i = self.word_index.get(w)
            if i is not None:
                if num_words and i >= num_words:
                    if oov_token_index is not None:
                        vect.append(oov_token_index)
                else:
                    vect.append(i)
            elif self.oov_token is not None:
                vect.append(oov_token_index)
        yield vect