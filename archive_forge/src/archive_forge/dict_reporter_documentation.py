import logging
import threading
from .metrics_reporter import AbstractMetricsReporter

        Return a string category for the metric.

        The category is made up of this reporter's prefix and the
        metric's group and tags.

        Examples:
            prefix = 'foo', group = 'bar', tags = {'a': 1, 'b': 2}
            returns: 'foo.bar.a=1,b=2'

            prefix = 'foo', group = 'bar', tags = None
            returns: 'foo.bar'

            prefix = None, group = 'bar', tags = None
            returns: 'bar'
        