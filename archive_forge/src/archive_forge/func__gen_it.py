from collections import abc
import functools
import itertools
def _gen_it(all_its):
    seen = set()
    for it in all_its:
        for value in it:
            if seen_selector is not None:
                maybe_seen_value = seen_selector(value)
            else:
                maybe_seen_value = value
            if maybe_seen_value not in seen:
                yield value
                seen.add(maybe_seen_value)