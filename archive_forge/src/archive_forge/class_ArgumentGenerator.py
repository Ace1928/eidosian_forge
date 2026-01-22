import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
class ArgumentGenerator:
    """Generate sample input based on a shape model.

    This class contains a ``generate_skeleton`` method that will take
    an input/output shape (created from ``botocore.model``) and generate
    a sample dictionary corresponding to the input/output shape.

    The specific values used are place holder values. For strings either an
    empty string or the member name can be used, for numbers 0 or 0.0 is used.
    The intended usage of this class is to generate the *shape* of the input
    structure.

    This can be useful for operations that have complex input shapes.
    This allows a user to just fill in the necessary data instead of
    worrying about the specific structure of the input arguments.

    Example usage::

        s = botocore.session.get_session()
        ddb = s.get_service_model('dynamodb')
        arg_gen = ArgumentGenerator()
        sample_input = arg_gen.generate_skeleton(
            ddb.operation_model('CreateTable').input_shape)
        print("Sample input for dynamodb.CreateTable: %s" % sample_input)

    """

    def __init__(self, use_member_names=False):
        self._use_member_names = use_member_names

    def generate_skeleton(self, shape):
        """Generate a sample input.

        :type shape: ``botocore.model.Shape``
        :param shape: The input shape.

        :return: The generated skeleton input corresponding to the
            provided input shape.

        """
        stack = []
        return self._generate_skeleton(shape, stack)

    def _generate_skeleton(self, shape, stack, name=''):
        stack.append(shape.name)
        try:
            if shape.type_name == 'structure':
                return self._generate_type_structure(shape, stack)
            elif shape.type_name == 'list':
                return self._generate_type_list(shape, stack)
            elif shape.type_name == 'map':
                return self._generate_type_map(shape, stack)
            elif shape.type_name == 'string':
                if self._use_member_names:
                    return name
                if shape.enum:
                    return random.choice(shape.enum)
                return ''
            elif shape.type_name in ['integer', 'long']:
                return 0
            elif shape.type_name in ['float', 'double']:
                return 0.0
            elif shape.type_name == 'boolean':
                return True
            elif shape.type_name == 'timestamp':
                return datetime.datetime(1970, 1, 1, 0, 0, 0)
        finally:
            stack.pop()

    def _generate_type_structure(self, shape, stack):
        if stack.count(shape.name) > 1:
            return {}
        skeleton = OrderedDict()
        for member_name, member_shape in shape.members.items():
            skeleton[member_name] = self._generate_skeleton(member_shape, stack, name=member_name)
        return skeleton

    def _generate_type_list(self, shape, stack):
        name = ''
        if self._use_member_names:
            name = shape.member.name
        return [self._generate_skeleton(shape.member, stack, name)]

    def _generate_type_map(self, shape, stack):
        key_shape = shape.key
        value_shape = shape.value
        assert key_shape.type_name == 'string'
        return OrderedDict([('KeyName', self._generate_skeleton(value_shape, stack))])