from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
Count the frequency of tokens in a column of strings.

    :class:`CountVectorizer` operates on columns that contain strings. For example:

    .. code-block::

                        corpus
        0    I dislike Python
        1       I like Python

    This preprocessors creates a column named like ``{column}_{token}`` for each
    unique token. These columns represent the frequency of token ``{token}`` in
    column ``{column}``. For example:

    .. code-block::

            corpus_I  corpus_Python  corpus_dislike  corpus_like
        0         1              1               1            0
        1         1              1               0            1

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import CountVectorizer
        >>>
        >>> df = pd.DataFrame({
        ...     "corpus": [
        ...         "Jimmy likes volleyball",
        ...         "Bob likes volleyball too",
        ...         "Bob also likes fruit jerky"
        ...     ]
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>>
        >>> vectorizer = CountVectorizer(["corpus"])
        >>> vectorizer.fit_transform(ds).to_pandas()  # doctest: +SKIP
           corpus_likes  corpus_volleyball  corpus_Bob  corpus_Jimmy  corpus_too  corpus_also  corpus_fruit  corpus_jerky
        0             1                  1           0             1           0            0             0             0
        1             1                  1           1             0           1            0             0             0
        2             1                  0           1             0           0            1             1             1

        You can limit the number of tokens in the vocabulary with ``max_features``.

        >>> vectorizer = CountVectorizer(["corpus"], max_features=3)
        >>> vectorizer.fit_transform(ds).to_pandas()  # doctest: +SKIP
           corpus_likes  corpus_volleyball  corpus_Bob
        0             1                  1           0
        1             1                  1           1
        2             1                  0           1

    Args:
        columns: The columns to separately tokenize and count.
        tokenization_fn: The function used to generate tokens. This function
            should accept a string as input and return a list of tokens as
            output. If unspecified, the tokenizer uses a function equivalent to
            ``lambda s: s.split(" ")``.
        max_features: The maximum number of tokens to encode in the transformed
            dataset. If specified, only the most frequent tokens are encoded.

    