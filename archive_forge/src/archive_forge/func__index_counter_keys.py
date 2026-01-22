import collections
from . import _constants as C
def _index_counter_keys(self, counter, unknown_token, reserved_tokens, most_freq_count, min_freq):
    """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `most_freq_count` and
        `min_freq`.
        """
    assert isinstance(counter, collections.Counter), '`counter` must be an instance of collections.Counter.'
    unknown_and_reserved_tokens = set(reserved_tokens) if reserved_tokens is not None else set()
    unknown_and_reserved_tokens.add(unknown_token)
    token_freqs = sorted(counter.items(), key=lambda x: x[0])
    token_freqs.sort(key=lambda x: x[1], reverse=True)
    token_cap = len(unknown_and_reserved_tokens) + (len(counter) if most_freq_count is None else most_freq_count)
    for token, freq in token_freqs:
        if freq < min_freq or len(self._idx_to_token) == token_cap:
            break
        if token not in unknown_and_reserved_tokens:
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._idx_to_token) - 1