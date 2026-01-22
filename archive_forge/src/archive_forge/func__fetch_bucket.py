import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def _fetch_bucket(self, bucket_name):
    """
        Method to return a bucket object which can be used to interact
        with an AWS S3 bucket using credentials found in a local file.

        Parameters
        ----------
        self : nipype.interfaces.io.DataSink
            self for instance method
        bucket_name : string
            string corresponding to the name of the bucket on S3

        Returns
        -------
        bucket : boto3.resources.factory.s3.Bucket
            boto3 s3 Bucket object which is used to interact with files
            in an S3 bucket on AWS
        """
    try:
        import boto3
        import botocore
    except ImportError as exc:
        err_msg = 'Boto3 package is not installed - install boto3 and try again.'
        raise Exception(err_msg)
    creds_path = self.inputs.creds_path
    try:
        aws_access_key_id, aws_secret_access_key = self._return_aws_keys()
    except Exception as exc:
        err_msg = 'There was a problem extracting the AWS credentials from the credentials file provided: %s. Error:\n%s' % (creds_path, exc)
        raise Exception(err_msg)
    if aws_access_key_id and aws_secret_access_key:
        iflogger.info('Connecting to S3 bucket: %s with credentials...', bucket_name)
        session = boto3.session.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    else:
        iflogger.info('Connecting to S3 bucket: %s with IAM role...', bucket_name)
        session = boto3.session.Session()
    s3_resource = session.resource('s3', use_ssl=True)
    try:
        _get_head_bucket(s3_resource, bucket_name)
    except Exception as exc:
        s3_resource.meta.client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)
        iflogger.info('Connecting to AWS: %s anonymously...', bucket_name)
        _get_head_bucket(s3_resource, bucket_name)
    bucket = s3_resource.Bucket(bucket_name)
    return bucket