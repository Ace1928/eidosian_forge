import collections
import random
import threading
def FilterItems(self, filterFn):
    """Filter items in a ReservoirBucket, using a filtering function.

        Filtering items from the reservoir bucket must update the
        internal state variable self._num_items_seen, which is used for determining
        the rate of replacement in reservoir sampling. Ideally, self._num_items_seen
        would contain the exact number of items that have ever seen by the
        ReservoirBucket and satisfy filterFn. However, the ReservoirBucket does not
        have access to all items seen -- it only has access to the subset of items
        that have survived sampling (self.items). Therefore, we estimate
        self._num_items_seen by scaling it by the same ratio as the ratio of items
        not removed from self.items.

        Args:
          filterFn: A function that returns True for items to be kept.

        Returns:
          The number of items removed from the bucket.
        """
    with self._mutex:
        size_before = len(self.items)
        self.items = list(filter(filterFn, self.items))
        size_diff = size_before - len(self.items)
        prop_remaining = len(self.items) / float(size_before) if size_before > 0 else 0
        self._num_items_seen = int(round(self._num_items_seen * prop_remaining))
        return size_diff