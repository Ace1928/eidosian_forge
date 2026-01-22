from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class CountVectorizer(Preprocessor):
    """Count the frequency of tokens in a column of strings.

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

    """

    def __init__(self, columns: List[str], tokenization_fn: Optional[Callable[[str], List[str]]]=None, max_features: Optional[int]=None):
        self.columns = columns
        self.tokenization_fn = tokenization_fn or simple_split_tokenizer
        self.max_features = max_features

    def _fit(self, dataset: Dataset) -> Preprocessor:

        def get_pd_value_counts(df: pd.DataFrame) -> List[Counter]:

            def get_token_counts(col):
                token_series = df[col].apply(self.tokenization_fn)
                tokens = token_series.sum()
                return Counter(tokens)
            return {col: [get_token_counts(col)] for col in self.columns}
        value_counts = dataset.map_batches(get_pd_value_counts, batch_format='pandas')
        total_counts = {col: Counter() for col in self.columns}
        for batch in value_counts.iter_batches(batch_size=None):
            for col, counters in batch.items():
                for counter in counters:
                    total_counts[col].update(counter)

        def most_common(counter: Counter, n: int):
            return Counter(dict(counter.most_common(n)))
        top_counts = [most_common(counter, self.max_features) for counter in total_counts.values()]
        self.stats_ = {f'token_counts({col})': counts for col, counts in zip(self.columns, top_counts)}
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        to_concat = []
        for col in self.columns:
            token_counts = self.stats_[f'token_counts({col})']
            sorted_tokens = [token for token, count in token_counts.most_common()]
            tokenized = df[col].map(self.tokenization_fn).map(Counter)
            for token in sorted_tokens:
                series = tokenized.map(lambda val: val[token])
                series.name = f'{col}_{token}'
                to_concat.append(series)
        df = pd.concat(to_concat, axis=1)
        return df

    def __repr__(self):
        fn_name = getattr(self.tokenization_fn, '__name__', self.tokenization_fn)
        return f'{self.__class__.__name__}(columns={self.columns!r}, tokenization_fn={fn_name}, max_features={self.max_features!r})'