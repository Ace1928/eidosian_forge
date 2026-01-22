from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateDeviceList(args, catalog_mgr):
    """Validates that --device contains a valid set of dimensions and values."""
    if not args.device:
        return
    for device_spec in args.device:
        for dim, val in device_spec.items():
            device_spec[dim] = catalog_mgr.ValidateDimensionAndValue(dim, val)
        if 'model' not in device_spec:
            device_spec['model'] = catalog_mgr.GetDefaultModel()
        if 'version' not in device_spec:
            device_spec['version'] = catalog_mgr.GetDefaultVersion()
        if 'locale' not in device_spec:
            device_spec['locale'] = catalog_mgr.GetDefaultLocale()
        if 'orientation' not in device_spec:
            device_spec['orientation'] = catalog_mgr.GetDefaultOrientation()