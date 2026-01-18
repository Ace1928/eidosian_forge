from math import ceil
from boto.compat import json, map, six
import requests
def to_params(self):
    """Transform search parameters from instance properties to a dictionary

        :rtype: dict
        :return: search parameters
        """
    params = {'start': self.start, 'size': self.real_size}
    if self.q:
        params['q'] = self.q
    if self.bq:
        params['bq'] = self.bq
    if self.rank:
        params['rank'] = ','.join(self.rank)
    if self.return_fields:
        params['return-fields'] = ','.join(self.return_fields)
    if self.facet:
        params['facet'] = ','.join(self.facet)
    if self.facet_constraints:
        for k, v in six.iteritems(self.facet_constraints):
            params['facet-%s-constraints' % k] = v
    if self.facet_sort:
        for k, v in six.iteritems(self.facet_sort):
            params['facet-%s-sort' % k] = v
    if self.facet_top_n:
        for k, v in six.iteritems(self.facet_top_n):
            params['facet-%s-top-n' % k] = v
    if self.t:
        for k, v in six.iteritems(self.t):
            params['t-%s' % k] = v
    return params