import abc
import threading
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_messaging import exceptions
class PollStyleListener(object, metaclass=abc.ABCMeta):
    """A PollStyleListener is used to transfer received messages to a server
    for processing. A polling pattern is used to retrieve messages.  A
    PollStyleListener uses a separate thread to run the polling loop.  A
    :py:class:`PollStyleListenerAdapter` can be used to create a
    :py:class:`Listener` from a PollStyleListener.

    :param prefetch_size: The number of messages that should be pulled from the
        backend per receive transaction. May not be honored by all backend
        implementations.
    :type prefetch_size: int
    """

    def __init__(self, prefetch_size=-1):
        self.prefetch_size = prefetch_size

    @abc.abstractmethod
    def poll(self, timeout=None, batch_size=1, batch_timeout=None):
        """poll is called by the server to retrieve incoming messages. It
        blocks until 'batch_size' incoming messages are available, a timeout
        occurs, or the poll is interrupted by a call to the :py:meth:`stop`
        method.

        If 'batch_size' is > 1 poll must block until 'batch_size' messages are
        available or at least one message is available and batch_timeout
        expires

        :param timeout: Block up to 'timeout' seconds waiting for a message
        :type timeout: float
        :param batch_size: Block until this number of messages are received.
        :type batch_size: int
        :param batch_timeout: Time to wait in seconds for a full batch to
            arrive. A timer is started when the first message in a batch is
            received. If a full batch's worth of messages is not received when
            the timer expires then :py:meth:`poll` returns all messages
            received thus far.
        :type batch_timeout: float
        :raises: Does not raise an exception.
        :return: A list of up to batch_size IncomingMessage objects.
        """

    def stop(self):
        """Stop the listener from polling for messages. This method must cause
        the :py:meth:`poll` call to unblock and return whatever messages are
        currently available.  This method is called from a different thread
        than the poller so it must be thread-safe.
        """
        pass

    def cleanup(self):
        """Cleanup all resources held by the listener. This method should block
        until the cleanup is completed.
        """
        pass