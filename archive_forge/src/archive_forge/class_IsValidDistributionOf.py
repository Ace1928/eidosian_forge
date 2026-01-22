import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class IsValidDistributionOf(object):
    """Test whether a given list can be split into particular
    sub-lists. All items in the original list must be in exactly one
    sub-list, and must appear in that sub-list in the same order with
    respect to any other items as in the original list.
    """

    def __init__(self, original):
        self.original = original

    def __str__(self):
        return 'IsValidDistribution(%s)' % self.original

    def match(self, actual):
        errors = InvalidDistribution(self.original, actual)
        received = [[idx for idx in act] for act in actual]

        def _remove(obj, lists):
            for li in lists:
                if obj in li:
                    front = li[0]
                    li.remove(obj)
                    return front
            return None
        for item in self.original:
            o = _remove(item, received)
            if not o:
                errors.missing += item
            elif item != o:
                errors.wrong_order.append([item, o])
        for li in received:
            errors.extra += li
        return errors or None