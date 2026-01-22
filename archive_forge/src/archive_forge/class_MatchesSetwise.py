from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
class MatchesSetwise:
    """Matches if all the matchers match elements of the value being matched.

    That is, each element in the 'observed' set must match exactly one matcher
    from the set of matchers, with no matchers left over.

    The difference compared to `MatchesListwise` is that the order of the
    matchings does not matter.
    """

    def __init__(self, *matchers):
        self.matchers = matchers

    def match(self, observed):
        remaining_matchers = set(self.matchers)
        not_matched = []
        for value in observed:
            for matcher in remaining_matchers:
                if matcher.match(value) is None:
                    remaining_matchers.remove(matcher)
                    break
            else:
                not_matched.append(value)
        if not_matched or remaining_matchers:
            remaining_matchers = list(remaining_matchers)
            if len(not_matched) == 0:
                if len(remaining_matchers) > 1:
                    msg = 'There were {} matchers left over: '.format(len(remaining_matchers))
                else:
                    msg = 'There was 1 matcher left over: '
                msg += ', '.join(map(str, remaining_matchers))
                return Mismatch(msg)
            elif len(remaining_matchers) == 0:
                if len(not_matched) > 1:
                    return Mismatch('There were {} values left over: {}'.format(len(not_matched), not_matched))
                else:
                    return Mismatch('There was 1 value left over: {}'.format(not_matched))
            else:
                common_length = min(len(remaining_matchers), len(not_matched))
                if common_length == 0:
                    raise AssertionError("common_length can't be 0 here")
                if common_length > 1:
                    msg = f'There were {common_length} mismatches'
                else:
                    msg = 'There was 1 mismatch'
                if len(remaining_matchers) > len(not_matched):
                    extra_matchers = remaining_matchers[common_length:]
                    msg += f' and {len(extra_matchers)} extra matcher'
                    if len(extra_matchers) > 1:
                        msg += 's'
                    msg += ': ' + ', '.join(map(str, extra_matchers))
                elif len(not_matched) > len(remaining_matchers):
                    extra_values = not_matched[common_length:]
                    msg += f' and {len(extra_values)} extra value'
                    if len(extra_values) > 1:
                        msg += 's'
                    msg += ': ' + str(extra_values)
                return Annotate(msg, MatchesListwise(remaining_matchers[:common_length])).match(not_matched[:common_length])