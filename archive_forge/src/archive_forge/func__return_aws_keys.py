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
def _return_aws_keys(self):
    """
        Method to return AWS access key id and secret access key using
        credentials found in a local file.

        Parameters
        ----------
        self : nipype.interfaces.io.DataSink
            self for instance method

        Returns
        -------
        aws_access_key_id : string
            string of the AWS access key ID
        aws_secret_access_key : string
            string of the AWS secret access key
        """
    import os
    creds_path = self.inputs.creds_path
    if creds_path and os.path.exists(creds_path):
        with open(creds_path, 'r') as creds_in:
            row1 = creds_in.readline()
            row2 = creds_in.readline()
        if 'User Name' in row1:
            aws_access_key_id = row2.split(',')[1]
            aws_secret_access_key = row2.split(',')[2]
        elif 'AWSAccessKeyId' in row1:
            aws_access_key_id = row1.split('=')[1]
            aws_secret_access_key = row2.split('=')[1]
        else:
            err_msg = 'Credentials file not recognized, check file is correct'
            raise Exception(err_msg)
        aws_access_key_id = aws_access_key_id.replace('\r', '').replace('\n', '')
        aws_secret_access_key = aws_secret_access_key.replace('\r', '').replace('\n', '')
    else:
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    return (aws_access_key_id, aws_secret_access_key)