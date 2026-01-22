import logging
import os
import time
from ray.util.debug import log_once
from ray.rllib.utils.framework import try_import_tf
class _TFRunBuilder:
    """Used to incrementally build up a TensorFlow run.

    This is particularly useful for batching ops from multiple different
    policies in the multi-agent setting.
    """

    def __init__(self, session, debug_name):
        self.session = session
        self.debug_name = debug_name
        self.feed_dict = {}
        self.fetches = []
        self._executed = None

    def add_feed_dict(self, feed_dict):
        assert not self._executed
        for k in feed_dict:
            if k in self.feed_dict:
                raise ValueError('Key added twice: {}'.format(k))
        self.feed_dict.update(feed_dict)

    def add_fetches(self, fetches):
        assert not self._executed
        base_index = len(self.fetches)
        self.fetches.extend(fetches)
        return list(range(base_index, len(self.fetches)))

    def get(self, to_fetch):
        if self._executed is None:
            try:
                self._executed = _run_timeline(self.session, self.fetches, self.debug_name, self.feed_dict, os.environ.get('TF_TIMELINE_DIR'))
            except Exception as e:
                logger.exception('Error fetching: {}, feed_dict={}'.format(self.fetches, self.feed_dict))
                raise e
        if isinstance(to_fetch, int):
            return self._executed[to_fetch]
        elif isinstance(to_fetch, list):
            return [self.get(x) for x in to_fetch]
        elif isinstance(to_fetch, tuple):
            return tuple((self.get(x) for x in to_fetch))
        else:
            raise ValueError('Unsupported fetch type: {}'.format(to_fetch))