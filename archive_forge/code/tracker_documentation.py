from __future__ import annotations
import time
from threading import Event
from zmq.backend import Frame
from zmq.error import NotDone
Wait for 0MQ to be done with the message or until `timeout`.

        Parameters
        ----------
        timeout : float
            default: -1, which means wait forever.
            Maximum time in (s) to wait before raising NotDone.

        Returns
        -------
        None
            if done before `timeout`

        Raises
        ------
        NotDone
            if `timeout` reached before I am done.
        