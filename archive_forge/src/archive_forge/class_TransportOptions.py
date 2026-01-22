import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class TransportOptions(object):

    def __init__(self, at_least_once=False):
        self._at_least_once = at_least_once

    @property
    def at_least_once(self):
        return self._at_least_once