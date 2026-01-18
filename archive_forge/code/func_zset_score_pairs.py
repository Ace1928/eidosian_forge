import datetime
from redis.utils import str_if_bytes
def zset_score_pairs(response, **options):
    """
    If ``withscores`` is specified in the options, return the response as
    a list of (value, score) pairs
    """
    if not response or not options.get('withscores'):
        return response
    score_cast_func = options.get('score_cast_func', float)
    it = iter(response)
    return list(zip(it, map(score_cast_func, it)))