import datetime
import enum
import logging
import socket
import sys
import threading
import msgpack
from oslo_privsep._i18n import _
from oslo_utils import uuidutils
class ClientChannel(object):

    def __init__(self, sock):
        self.running = False
        self.writer = Serializer(sock)
        self.lock = threading.Lock()
        self.reader_thread = threading.Thread(name='privsep_reader', target=self._reader_main, args=(Deserializer(sock),))
        self.reader_thread.daemon = True
        self.outstanding_msgs = {}
        self.reader_thread.start()

    def _reader_main(self, reader):
        """This thread owns and demuxes the read channel"""
        with self.lock:
            self.running = True
        for msg in reader:
            msgid, data = msg
            if msgid is None:
                self.out_of_band(data)
            else:
                with self.lock:
                    if msgid not in self.outstanding_msgs:
                        LOG.warning('msgid should be in oustanding_msgs, it ispossible that timeout is reached!')
                        continue
                    self.outstanding_msgs[msgid].set_result(data)
        LOG.debug('EOF on privsep read channel')
        exc = IOError(_('Premature eof waiting for privileged process'))
        with self.lock:
            for mbox in self.outstanding_msgs.values():
                mbox.set_exception(exc)
            self.running = False

    def out_of_band(self, msg):
        """Received OOB message. Subclasses might want to override this."""
        pass

    def send_recv(self, msg, timeout=None):
        myid = uuidutils.generate_uuid()
        while myid in self.outstanding_msgs:
            LOG.warning("myid shoudn't be in outstanding_msgs.")
            myid = uuidutils.generate_uuid()
        future = Future(self.lock, timeout)
        with self.lock:
            self.outstanding_msgs[myid] = future
            try:
                self.writer.send((myid, msg))
                reply = future.result()
            except Exception:
                LOG.warning('Unexpected error: {}'.format(sys.exc_info()[0]))
                raise
            finally:
                del self.outstanding_msgs[myid]
        return reply

    def close(self):
        with self.lock:
            self.writer.close()
        self.reader_thread.join()