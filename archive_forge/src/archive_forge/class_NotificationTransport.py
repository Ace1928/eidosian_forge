import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class NotificationTransport(Transport):
    """Transport object for notifications."""

    def __init__(self, driver):
        super(NotificationTransport, self).__init__(driver)