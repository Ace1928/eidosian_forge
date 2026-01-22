import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCEList:
    """
    An Iterator that wraps list functions to provide additional features.

    GCE enforces a limit on the number of objects returned by a list operation,
    so users with more than 500 objects of a particular type will need to use
    filter(), page() or both.

    >>> l=GCEList(driver, driver.ex_list_urlmaps)
    >>> for sublist in l.filter('name eq ...-map').page(1):
    ...   sublist
    ...
    [<GCEUrlMap id="..." name="cli-map">]
    [<GCEUrlMap id="..." name="web-map">]

    One can create a GCEList manually, but it's slightly easier to use the
    ex_list() method of :class:`GCENodeDriver`.
    """

    def __init__(self, driver, list_fn, **kwargs):
        """
        :param  driver: An initialized :class:``GCENodeDriver``
        :type   driver: :class:``GCENodeDriver``

        :param  list_fn: A bound list method from :class:`GCENodeDriver`.
        :type   list_fn: ``instancemethod``
        """
        self.driver = driver
        self.list_fn = list_fn
        self.kwargs = kwargs
        self.params = {}

    def __iter__(self):
        list_fn = self.list_fn
        more_results = True
        while more_results:
            self.driver.connection.gce_params = self.params
            yield list_fn(**self.kwargs)
            more_results = 'pageToken' in self.params

    def __repr__(self):
        return '<GCEList list="{}" params="{}">'.format(self.list_fn.__name__, repr(self.params))

    def filter(self, expression):
        """
        Filter results of a list operation.

        GCE supports server-side filtering of resources returned by a list
        operation. Syntax of the filter expression is fully described in the
        GCE API reference doc, but in brief it is::

            FIELD_NAME COMPARISON_STRING LITERAL_STRING

        where FIELD_NAME is the resource's property name, COMPARISON_STRING is
        'eq' or 'ne', and LITERAL_STRING is a regular expression in RE2 syntax.

        >>> for sublist in l.filter('name eq ...-map'):
        ...   sublist
        ...
        [<GCEUrlMap id="..." name="cli-map">,                 <GCEUrlMap id="..." name="web-map">]

        API reference: https://cloud.google.com/compute/docs/reference/latest/
        RE2 syntax: https://github.com/google/re2/blob/master/doc/syntax.txt

        :param  expression: Filter expression described above.
        :type   expression: ``str``

        :return: This :class:`GCEList` instance
        :rtype:  :class:`GCEList`
        """
        self.params['filter'] = expression
        return self

    def page(self, max_results=500):
        """
        Limit the number of results by each iteration.

        This implements the paging functionality of the GCE list methods and
        returns this GCEList instance so that results can be chained:

        >>> for sublist in GCEList(driver, driver.ex_list_urlmaps).page(2):
        ...   sublist
        ...
        [<GCEUrlMap id="..." name="cli-map">,                 <GCEUrlMap id="..." name="lc-map">]
        [<GCEUrlMap id="..." name="web-map">]

        :keyword  max_results: Maximum number of results to return per
                               iteration. Defaults to the GCE default of 500.
        :type     max_results: ``int``

        :return: This :class:`GCEList` instance
        :rtype:  :class:`GCEList`
        """
        self.params['maxResults'] = max_results
        return self