from __future__ import annotations
import traceback
from typing import Any, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.i18n import _
class BrickException(Exception):
    """Base Brick Exception

    To correctly use this class, inherit from it and define
    a 'message' property. That message will get printf'd
    with the keyword arguments provided to the constructor.
    """
    message = _('An unknown exception occurred.')
    code = 500
    headers: dict = {}
    safe = False

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if 'code' not in self.kwargs:
            try:
                self.kwargs['code'] = self.code
            except AttributeError:
                pass
        if not message:
            try:
                message = self.message % kwargs
            except Exception:
                LOG.exception("Exception in string format operation. msg='%s'", self.message)
                for name, value in kwargs.items():
                    LOG.error('%(name)s: %(value)s', {'name': name, 'value': value})
                message = self.message
        self.msg = message
        super(BrickException, self).__init__(message)