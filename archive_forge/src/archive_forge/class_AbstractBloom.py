from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
class AbstractBloom(object):
    """
    The client allows to interact with RedisBloom and use all of
    it's functionality.

    - BF for Bloom Filter
    - CF for Cuckoo Filter
    - CMS for Count-Min Sketch
    - TOPK for TopK Data Structure
    - TDIGEST for estimate rank statistics
    """

    @staticmethod
    def append_items(params, items):
        """Append ITEMS to params."""
        params.extend(['ITEMS'])
        params += items

    @staticmethod
    def append_error(params, error):
        """Append ERROR to params."""
        if error is not None:
            params.extend(['ERROR', error])

    @staticmethod
    def append_capacity(params, capacity):
        """Append CAPACITY to params."""
        if capacity is not None:
            params.extend(['CAPACITY', capacity])

    @staticmethod
    def append_expansion(params, expansion):
        """Append EXPANSION to params."""
        if expansion is not None:
            params.extend(['EXPANSION', expansion])

    @staticmethod
    def append_no_scale(params, noScale):
        """Append NONSCALING tag to params."""
        if noScale is not None:
            params.extend(['NONSCALING'])

    @staticmethod
    def append_weights(params, weights):
        """Append WEIGHTS to params."""
        if len(weights) > 0:
            params.append('WEIGHTS')
            params += weights

    @staticmethod
    def append_no_create(params, noCreate):
        """Append NOCREATE tag to params."""
        if noCreate is not None:
            params.extend(['NOCREATE'])

    @staticmethod
    def append_items_and_increments(params, items, increments):
        """Append pairs of items and increments to params."""
        for i in range(len(items)):
            params.append(items[i])
            params.append(increments[i])

    @staticmethod
    def append_values_and_weights(params, items, weights):
        """Append pairs of items and weights to params."""
        for i in range(len(items)):
            params.append(items[i])
            params.append(weights[i])

    @staticmethod
    def append_max_iterations(params, max_iterations):
        """Append MAXITERATIONS to params."""
        if max_iterations is not None:
            params.extend(['MAXITERATIONS', max_iterations])

    @staticmethod
    def append_bucket_size(params, bucket_size):
        """Append BUCKETSIZE to params."""
        if bucket_size is not None:
            params.extend(['BUCKETSIZE', bucket_size])