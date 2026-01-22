import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def IsCaptureContext(context):
    return context.get('labels', {}).get('category', None) == CAPTURE_CATEGORY