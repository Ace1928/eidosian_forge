from boto.pyami.config import Config
from boto.services.message import ServiceMessage
import boto

        Returns the AWS object associated with a given option.

        The heuristics used are a bit lame.  If the option name contains
        the word 'bucket' it is assumed to be an S3 bucket, if the name
        contains the word 'queue' it is assumed to be an SQS queue and
        if it contains the word 'domain' it is assumed to be a SimpleDB
        domain.  If the option name specified does not exist in the
        config file or if the AWS object cannot be retrieved this
        returns None.
        