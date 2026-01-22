from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import encoding
import requests
class SurpriseError(Exception):
    pass