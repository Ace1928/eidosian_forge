import typing as t
from enum import Enum, auto

    Checks whether a key is in a trie.

    Examples:
        >>> in_trie(new_trie(["cat"]), "bob")
        (<TrieResult.FAILED: 1>, {'c': {'a': {'t': {0: True}}}})

        >>> in_trie(new_trie(["cat"]), "ca")
        (<TrieResult.PREFIX: 2>, {'t': {0: True}})

        >>> in_trie(new_trie(["cat"]), "cat")
        (<TrieResult.EXISTS: 3>, {0: True})

    Args:
        trie: The trie to be searched.
        key: The target key.

    Returns:
        A pair `(value, subtrie)`, where `subtrie` is the sub-trie we get at the point
        where the search stops, and `value` is a TrieResult value that can be one of:

        - TrieResult.FAILED: the search was unsuccessful
        - TrieResult.PREFIX: `value` is a prefix of a keyword in `trie`
        - TrieResult.EXISTS: `key` exists in `trie`
    