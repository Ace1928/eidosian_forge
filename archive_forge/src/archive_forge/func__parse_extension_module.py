import abc
import copy
from http import client as http_client
from urllib import parse as urlparse
from oslo_utils import strutils
from ironicclient.common.apiclient import exceptions
from ironicclient.common.i18n import _
def _parse_extension_module(self):
    self.manager_class = None
    for attr_name, attr_value in self.module.__dict__.items():
        if attr_name in self.SUPPORTED_HOOKS:
            self.add_hook(attr_name, attr_value)
        else:
            try:
                if issubclass(attr_value, BaseManager):
                    self.manager_class = attr_value
            except TypeError:
                pass