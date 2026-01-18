from math import ceil
from boto.compat import json, map, six
import requests
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def to_domain_connection_params(self):
    """
        Transform search parameters from instance properties to a dictionary
        that CloudSearchDomainConnection can accept

        :rtype: dict
        :return: search parameters
        """
    params = {'start': self.start, 'size': self.real_size}
    if self.q:
        params['q'] = self.q
    if self.parser:
        params['query_parser'] = self.parser
    if self.fq:
        params['filter_query'] = self.fq
    if self.expr:
        expr = {}
        for k, v in six.iteritems(self.expr):
            expr['expr.%s' % k] = v
        params['expr'] = expr
    if self.facet:
        facet = {}
        for k, v in six.iteritems(self.facet):
            if not isinstance(v, six.string_types):
                v = json.dumps(v)
            facet['facet.%s' % k] = v
        params['facet'] = facet
    if self.highlight:
        highlight = {}
        for k, v in six.iteritems(self.highlight):
            highlight['highlight.%s' % k] = v
        params['highlight'] = highlight
    if self.options:
        params['query_options'] = self.options
    if self.return_fields:
        params['ret'] = ','.join(self.return_fields)
    if self.partial is not None:
        params['partial'] = self.partial
    if self.sort:
        params['sort'] = ','.join(self.sort)
    return params