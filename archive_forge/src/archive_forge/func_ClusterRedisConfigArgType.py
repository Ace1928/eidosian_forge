from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def ClusterRedisConfigArgType(value):
    return arg_parsers.ArgDict()(value)