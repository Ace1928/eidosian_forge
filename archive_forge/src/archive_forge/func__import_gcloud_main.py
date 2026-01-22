from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import os
import sys
def _import_gcloud_main():
    """Returns reference to gcloud_main module."""
    import googlecloudsdk.gcloud_main
    return googlecloudsdk.gcloud_main