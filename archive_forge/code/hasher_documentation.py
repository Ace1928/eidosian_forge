import collections
from typing import List
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash
from ray.util.annotations import PublicAPI
Apply the `hashing trick <https://en.wikipedia.org/wiki/Feature_hashing>`_ to a
    table that describes token frequencies.

    :class:`FeatureHasher` creates ``num_features`` columns named ``hash_{index}``,
    where ``index`` ranges from :math:`0` to ``num_features``:math:`- 1`. The column
    ``hash_{index}`` describes the frequency of tokens that hash to ``index``.

    Distinct tokens can correspond to the same index. However, if ``num_features`` is
    large enough, then columns probably correspond to a unique token.

    This preprocessor is memory efficient and quick to pickle. However, given a
    transformed column, you can't know which tokens correspond to it. This might make it
    hard to determine which tokens are important to your model.

    .. warning::
        Sparse matrices aren't supported. If you use a large ``num_features``, this
        preprocessor might behave poorly.

    Examples:

        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import FeatureHasher

        The data below describes the frequencies of tokens in ``"I like Python"`` and
        ``"I dislike Python"``.

        >>> df = pd.DataFrame({
        ...     "I": [1, 1],
        ...     "like": [1, 0],
        ...     "dislike": [0, 1],
        ...     "Python": [1, 1]
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP

        :class:`FeatureHasher` hashes each token to determine its index. For example,
        the index of ``"I"`` is :math:`hash(\texttt{"I"}) \pmod 8 = 5`.

        >>> hasher = FeatureHasher(columns=["I", "like", "dislike", "Python"], num_features=8)
        >>> hasher.fit_transform(ds).to_pandas().to_numpy()  # doctest: +SKIP
        array([[0, 0, 0, 2, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 1, 1, 0]])

        Notice the hash collision: both ``"like"`` and ``"Python"`` correspond to index
        :math:`3`. You can avoid hash collisions like these by increasing
        ``num_features``.

    Args:
        columns: The columns to apply the hashing trick to. Each column should describe
            the frequency of a token.
        num_features: The number of features used to represent the vocabulary. You
            should choose a value large enough to prevent hash collisions between
            distinct tokens.

    .. seealso::
        :class:`~ray.data.preprocessors.CountVectorizer`
            Use this preprocessor to generate inputs for :class:`FeatureHasher`.

        :class:`ray.data.preprocessors.HashingVectorizer`
            If your input data describes documents rather than token frequencies,
            use :class:`~ray.data.preprocessors.HashingVectorizer`.
    