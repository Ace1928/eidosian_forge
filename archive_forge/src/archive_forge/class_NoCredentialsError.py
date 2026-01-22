from __future__ import print_function
import os
import urlparse
import boto
import boto.connection
import boto.jsonresponse
import boto.exception
from boto.roboto import awsqueryrequest
class NoCredentialsError(boto.exception.BotoClientError):

    def __init__(self):
        s = 'Unable to find credentials'
        super(NoCredentialsError, self).__init__(s)