import argparse
import functools
import hashlib
import logging
import os
from oslo_utils import encodeutils
from oslo_utils import importutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
def _safe_encode_without_obj(data):
    if isinstance(data, str):
        return encodeutils.safe_encode(data)
    return data