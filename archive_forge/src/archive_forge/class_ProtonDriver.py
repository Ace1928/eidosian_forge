import collections
import logging
import os
import threading
import uuid
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_messaging.target import Target
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_messaging._drivers.amqp1_driver.eventloop import compute_timeout
from oslo_messaging._drivers.amqp1_driver import opts
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common
@removals.removed_class('ProtonDriver')
class ProtonDriver(base.BaseDriver):
    """AMQP 1.0 Driver

    See :doc:`AMQP1.0` for details.
    """

    def __init__(self, conf, url, default_exchange=None, allowed_remote_exmods=[]):
        if proton is None or controller is None:
            raise NotImplementedError('Proton AMQP C libraries not installed')
        super(ProtonDriver, self).__init__(conf, url, default_exchange, allowed_remote_exmods)
        opt_group = cfg.OptGroup(name='oslo_messaging_amqp', title='AMQP 1.0 driver options')
        conf.register_group(opt_group)
        conf.register_opts(opts.amqp1_opts, group=opt_group)
        conf = common.ConfigOptsProxy(conf, url, opt_group.name)
        self._conf = conf
        self._default_exchange = default_exchange
        self._ctrl = None
        self._pid = None
        self._lock = threading.Lock()
        opt_name = conf.oslo_messaging_amqp
        self._default_reply_timeout = opt_name.default_reply_timeout
        self._default_send_timeout = opt_name.default_send_timeout
        self._default_notify_timeout = opt_name.default_notify_timeout
        self._default_reply_retry = opt_name.default_reply_retry
        ps = [s.lower() for s in opt_name.pre_settled]
        self._pre_settle_call = 'rpc-call' in ps
        self._pre_settle_reply = 'rpc-reply' in ps
        self._pre_settle_cast = 'rpc-cast' in ps
        self._pre_settle_notify = 'notify' in ps
        bad_opts = set(ps).difference(['rpc-call', 'rpc-reply', 'rpc-cast', 'notify'])
        if bad_opts:
            LOG.warning('Ignoring unrecognized pre_settle value(s): %s', ' '.join(bad_opts))

    def _ensure_connect_called(func):
        """Causes a new controller to be created when the messaging service is
        first used by the current process. It is safe to push tasks to it
        whether connected or not, but those tasks won't be processed until
        connection completes.
        """

        def wrap(self, *args, **kws):
            with self._lock:
                old_pid = self._pid
                self._pid = os.getpid()
                if old_pid != self._pid:
                    if self._ctrl is not None:
                        LOG.warning('Process forked after connection established!')
                        self._ctrl = None
                    self._ctrl = controller.Controller(self._url, self._default_exchange, self._conf)
                    self._ctrl.connect()
            return func(self, *args, **kws)
        return wrap

    @_ensure_connect_called
    def send(self, target, ctxt, message, wait_for_reply=False, timeout=None, call_monitor_timeout=None, retry=None, transport_options=None):
        """Send a message to the given target.

        :param target: destination for message
        :type target: oslo_messaging.Target
        :param ctxt: message context
        :type ctxt: dict
        :param message: message payload
        :type message: dict
        :param wait_for_reply: expects a reply message, wait for it
        :type wait_for_reply: bool
        :param timeout: raise exception if send does not complete within
                        timeout seconds. None == no timeout.
        :type timeout: float
        :param call_monitor_timeout: Maximum time the client will wait for the
            call to complete or receive a message heartbeat indicating the
            remote side is still executing.
        :type call_monitor_timeout: float
        :param retry: (optional) maximum re-send attempts on recoverable error
                      None or -1 means to retry forever
                      0 means no retry
                      N means N retries
        :type retry: int
        :param transport_options: transport-specific options to apply to the
                                  sending of the message (TBD)
        :type transport_options: dictionary
        """
        request = marshal_request(message, ctxt, None, call_monitor_timeout)
        if timeout:
            expire = compute_timeout(timeout)
            request.ttl = timeout
            request.expiry_time = compute_timeout(timeout)
        else:
            expire = compute_timeout(self._default_send_timeout)
        if wait_for_reply:
            ack = not self._pre_settle_call
            if call_monitor_timeout is None:
                task = controller.RPCCallTask(target, request, expire, retry, wait_for_ack=ack)
            else:
                task = controller.RPCMonitoredCallTask(target, request, expire, call_monitor_timeout, retry, wait_for_ack=ack)
        else:
            ack = not self._pre_settle_cast
            task = controller.SendTask('RPC Cast', request, target, expire, retry, wait_for_ack=ack)
        self._ctrl.add_task(task)
        reply = task.wait()
        if isinstance(reply, Exception):
            raise reply
        if reply:
            reply = unmarshal_response(reply, self._allowed_remote_exmods)
        return reply

    @_ensure_connect_called
    def send_notification(self, target, ctxt, message, version, retry=None):
        """Send a notification message to the given target.

        :param target: destination for message
        :type target: oslo_messaging.Target
        :param ctxt: message context
        :type ctxt: dict
        :param message: message payload
        :type message: dict
        :param version: message envelope version
        :type version: float
        :param retry: (optional) maximum re-send attempts on recoverable error
                      None or -1 means to retry forever
                      0 means no retry
                      N means N retries
        :type retry: int
        """
        request = marshal_request(message, ctxt, envelope=version == 2.0)
        deadline = compute_timeout(self._default_notify_timeout)
        ack = not self._pre_settle_notify
        task = controller.SendTask('Notify', request, target, deadline, retry, wait_for_ack=ack, notification=True)
        self._ctrl.add_task(task)
        rc = task.wait()
        if isinstance(rc, Exception):
            raise rc

    @_ensure_connect_called
    def listen(self, target, batch_size, batch_timeout):
        """Construct a Listener for the given target."""
        LOG.debug('Listen to %s', target)
        listener = ProtonListener(self)
        task = controller.SubscribeTask(target, listener)
        self._ctrl.add_task(task)
        task.wait()
        return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)

    @_ensure_connect_called
    def listen_for_notifications(self, targets_and_priorities, pool, batch_size, batch_timeout):
        """Construct a Listener for notifications on the given target and
        priority.
        """
        LOG.debug('Listen for notifications %s', targets_and_priorities)
        if pool:
            raise NotImplementedError('"pool" not implemented by this transport driver')
        listener = ProtonListener(self)
        for target, priority in targets_and_priorities:
            topic = '%s.%s' % (target.topic, priority)
            task = controller.SubscribeTask(Target(topic=topic), listener, notifications=True)
            self._ctrl.add_task(task)
            task.wait()
        return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)

    def cleanup(self):
        """Release all resources."""
        if self._ctrl:
            self._ctrl.shutdown()
            self._ctrl = None
        LOG.info('AMQP 1.0 messaging driver shutdown')

    def require_features(self, requeue=True):
        pass