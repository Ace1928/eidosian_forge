import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
class EncoderError(boto.exception.BotoClientError):

    def __init__(self, error_msg):
        s = 'Error encoding value (%s)' % error_msg
        super(EncoderError, self).__init__(s)