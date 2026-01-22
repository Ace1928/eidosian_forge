import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class EC2SubnetAssociation:
    """
    Class which stores information about Route Table associated with
    a given Subnet in a VPC

    Note: This class is VPC specific.
    """

    def __init__(self, id, route_table_id, subnet_id, main=False):
        """
        :param      id: The ID of the subnet association in the VPC.
        :type       id: ``str``

        :param      route_table_id: The ID of a route table in the VPC.
        :type       route_table_id: ``str``

        :param      subnet_id: The ID of a subnet in the VPC.
        :type       subnet_id: ``str``

        :param      main: If true, means this is a main VPC route table.
        :type       main: ``bool``
        """
        self.id = id
        self.route_table_id = route_table_id
        self.subnet_id = subnet_id
        self.main = main

    def __repr__(self):
        return '<EC2SubnetAssociation: id=%s>' % self.id