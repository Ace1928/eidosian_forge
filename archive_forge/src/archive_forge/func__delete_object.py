import xml.sax
import time
import boto
from boto.connection import AWSAuthConnection
from boto import handler
from boto.cloudfront.distribution import Distribution, DistributionSummary, DistributionConfig
from boto.cloudfront.distribution import StreamingDistribution, StreamingDistributionSummary, StreamingDistributionConfig
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.identity import OriginAccessIdentitySummary
from boto.cloudfront.identity import OriginAccessIdentityConfig
from boto.cloudfront.invalidation import InvalidationBatch, InvalidationSummary, InvalidationListResultSet
from boto.resultset import ResultSet
from boto.cloudfront.exception import CloudFrontServerError
def _delete_object(self, id, etag, resource):
    uri = '/%s/%s/%s' % (self.Version, resource, id)
    response = self.make_request('DELETE', uri, {'If-Match': etag})
    body = response.read()
    boto.log.debug(body)
    if response.status != 204:
        raise CloudFrontServerError(response.status, response.reason, body)