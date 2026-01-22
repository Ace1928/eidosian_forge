import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigmaSubscription:
    """
    Represents CloudSigma subscription.
    """

    def __init__(self, id, resource, amount, period, status, price, start_time, end_time, auto_renew, subscribed_object=None):
        """
        :param id: Subscription ID.
        :type id: ``str``

        :param resource: Resource (e.g vlan, ip, etc.).
        :type resource: ``str``

        :param period: Subscription period.
        :type period: ``str``

        :param status: Subscription status (active / inactive).
        :type status: ``str``

        :param price: Subscription price.
        :type price: ``str``

        :param start_time: Start time for this subscription.
        :type start_time: ``datetime.datetime``

        :param end_time: End time for this subscription.
        :type end_time: ``datetime.datetime``

        :param auto_renew: True if the subscription is auto renewed.
        :type auto_renew: ``bool``

        :param subscribed_object: Optional UUID of the subscribed object.
        :type subscribed_object: ``str``
        """
        self.id = id
        self.resource = resource
        self.amount = amount
        self.period = period
        self.status = status
        self.price = price
        self.start_time = start_time
        self.end_time = end_time
        self.auto_renew = auto_renew
        self.subscribed_object = subscribed_object

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<CloudSigmaSubscription id=%s, resource=%s, amount=%s, period=%s, object_uuid=%s>' % (self.id, self.resource, self.amount, self.period, self.subscribed_object)