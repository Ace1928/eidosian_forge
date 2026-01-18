import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.preprocessing.text.text_to_word_sequence')
def text_to_word_sequence(input_text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
    """Converts a text to a sequence of words (or tokens).

    Deprecated: `tf.keras.preprocessing.text.text_to_word_sequence` does not
    operate on tensors and is not recommended for new code. Prefer
    `tf.strings.regex_replace` and `tf.strings.split` which provide equivalent
    functionality and accept `tf.Tensor` input. For an overview of text handling
    in Tensorflow, see the [text loading tutorial]
    (https://www.tensorflow.org/tutorials/load_data/text).

    This function transforms a string of text into a list of words
    while ignoring `filters` which include punctuations by default.

    >>> sample_text = 'This is a sample sentence.'
    >>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
    ['this', 'is', 'a', 'sample', 'sentence']

    Args:
        input_text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\\\]^_`{|}~\\\\t\\\\n'``,
              includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    Returns:
        A list of words (or tokens).
    """
    if lower:
        input_text = input_text.lower()
    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)
    seq = input_text.split(split)
    return [i for i in seq if i]