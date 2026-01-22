from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core.util import files
def ValidateCreateJobArguments(args):
    """Valid parameters for create job command."""
    missing = None
    if args.file is None and args.json is None:
        input_uri = args.input_uri
        output_uri = args.output_uri
        if input_uri is None:
            missing = 'input-uri'
        elif output_uri is None:
            missing = 'output-uri'
    if missing is not None:
        raise calliope_exceptions.RequiredArgumentException('--{}'.format(missing), '{} is required when using template to create job'.format(missing))