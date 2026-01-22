from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc.jobs import base as job_base
from googlecloudsdk.command_lib.dataproc.jobs import util as job_util
@staticmethod
def GetFilesByType(args):
    """Returns a dict of files by their type (jars, archives, etc.)."""
    return {'main_jar': args.main_jar, 'jars': args.jars, 'archives': args.archives, 'files': args.files}