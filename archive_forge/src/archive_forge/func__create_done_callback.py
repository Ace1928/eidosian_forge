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
def _create_done_callback(self, msgid):
    """Creates a future callback to receive command execution results.

        :param msgid: The message identifier.
        :return: A future reply callback.
        """
    channel = self.channel

    def _call_back(result):
        """Future execution callback.

            :param result: The `future` execution and its results.
            """
        try:
            reply = result.result()
            LOG.debug('privsep: reply[%(msgid)s]: %(reply)s', {'msgid': msgid, 'reply': reply})
            channel.send((msgid, reply))
        except IOError:
            self.communication_error = sys.exc_info()
        except Exception as e:
            LOG.debug('privsep: Exception during request[%(msgid)s]: %(err)s', {'msgid': msgid, 'err': e}, exc_info=True)
            cls = e.__class__
            cls_name = '%s.%s' % (cls.__module__, cls.__name__)
            reply = (comm.Message.ERR.value, cls_name, e.args)
            try:
                channel.send((msgid, reply))
            except IOError as exc:
                self.communication_error = exc
    return _call_back