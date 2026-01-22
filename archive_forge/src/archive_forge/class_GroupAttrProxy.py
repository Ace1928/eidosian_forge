from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class GroupAttrProxy(Mapping):
    """Internal helper proxy for oslo_config.cfg.ConfigOpts.GroupAttr."""
    _VOID_MARKER = object()

    def __init__(self, conf, group_name, group, url):
        self._conf = conf
        self._group_name = group_name
        self._group = group
        self._url = url

    def __getattr__(self, opt_name):
        opt_value_conf = getattr(self._group, opt_name)
        opt_value_url = self._url.query.get(opt_name, self._VOID_MARKER)
        if opt_value_url is self._VOID_MARKER:
            return opt_value_conf
        opt_info = self._conf._get_opt_info(opt_name, self._group_name)
        return opt_info['opt'].type(opt_value_url)

    def __getitem__(self, opt_name):
        return self.__getattr__(opt_name)

    def __contains__(self, opt_name):
        return opt_name in self._group

    def __iter__(self):
        return iter(self._group)

    def __len__(self):
        return len(self._group)