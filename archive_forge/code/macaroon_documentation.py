from __future__ import unicode_literals
import copy
from base64 import standard_b64encode
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.serializers.binary_serializer import BinarySerializer
from pymacaroons.exceptions import MacaroonInitException
from pymacaroons.utils import (
from pymacaroons.caveat_delegates import (
 Return a new discharge macaroon bound to the receiving macaroon's
        current signature so that it can be used in a request.

        This must be done before a discharge macaroon is sent to a server.

        :param discharge_macaroon:
        :return: bound discharge macaroon
        