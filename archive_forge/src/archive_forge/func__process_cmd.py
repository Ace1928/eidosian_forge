from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def _process_cmd(self, msgid, cmd, *args):
    """Executes the requested command in an execution thread.

        This executes a call within a thread executor and returns the results
        of the execution.

        :param msgid: The message identifier.
        :param cmd: The `Message` type indicating the command type.
        :param args: The function, args, and kwargs if a Message.CALL type.
        :return: A tuple of the return status, optional call output, and
                 optional error information.
        """
    if cmd == comm.Message.PING:
        return (comm.Message.PONG.value,)
    try:
        if cmd != comm.Message.CALL:
            raise ProtocolError(_('Unknown privsep cmd: %s') % cmd)
        name, f_args, f_kwargs = args
        func = importutils.import_class(name)
        if not self.context.is_entrypoint(func):
            msg = _('Invalid privsep function: %s not exported') % name
            raise NameError(msg)
        ret = func(*f_args, **f_kwargs)
        return (comm.Message.RET.value, ret)
    except Exception as e:
        LOG.debug('privsep: Exception during request[%(msgid)s]: %(err)s', {'msgid': msgid, 'err': e}, exc_info=True)
        cls = e.__class__
        cls_name = '%s.%s' % (cls.__module__, cls.__name__)
        return (comm.Message.ERR.value, cls_name, e.args)