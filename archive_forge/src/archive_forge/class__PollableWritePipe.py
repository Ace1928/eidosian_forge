from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
@implementer(IConsumer)
class _PollableWritePipe(_PollableResource):

    def __init__(self, writePipe, lostCallback):
        self.disconnecting = False
        self.producer = None
        self.producerPaused = False
        self.streamingProducer = 0
        self.outQueue = []
        self.writePipe = writePipe
        self.lostCallback = lostCallback
        try:
            win32pipe.SetNamedPipeHandleState(writePipe, win32pipe.PIPE_NOWAIT, None, None)
        except pywintypes.error:
            pass

    def close(self):
        self.disconnecting = True

    def bufferFull(self):
        if self.producer is not None:
            self.producerPaused = True
            self.producer.pauseProducing()

    def bufferEmpty(self):
        if self.producer is not None and (not self.streamingProducer or self.producerPaused):
            self.producer.producerPaused = False
            self.producer.resumeProducing()
            return True
        return False

    def registerProducer(self, producer, streaming):
        """Register to receive data from a producer.

        This sets this selectable to be a consumer for a producer.  When this
        selectable runs out of data on a write() call, it will ask the producer
        to resumeProducing(). A producer should implement the IProducer
        interface.

        FileDescriptor provides some infrastructure for producer methods.
        """
        if self.producer is not None:
            raise RuntimeError('Cannot register producer %s, because producer %s was never unregistered.' % (producer, self.producer))
        if not self.active:
            producer.stopProducing()
        else:
            self.producer = producer
            self.streamingProducer = streaming
            if not streaming:
                producer.resumeProducing()

    def unregisterProducer(self):
        """Stop consuming data from a producer, without disconnecting."""
        self.producer = None

    def writeConnectionLost(self):
        self.deactivate()
        try:
            win32api.CloseHandle(self.writePipe)
        except pywintypes.error:
            pass
        self.lostCallback()

    def writeSequence(self, seq):
        """
        Append a C{list} or C{tuple} of bytes to the output buffer.

        @param seq: C{list} or C{tuple} of C{str} instances to be appended to
            the output buffer.

        @raise TypeError: If C{seq} contains C{unicode}.
        """
        if str in map(type, seq):
            raise TypeError('Unicode not allowed in output buffer.')
        self.outQueue.extend(seq)

    def write(self, data):
        """
        Append some bytes to the output buffer.

        @param data: C{str} to be appended to the output buffer.
        @type data: C{str}.

        @raise TypeError: If C{data} is C{unicode} instead of C{str}.
        """
        if isinstance(data, str):
            raise TypeError('Unicode not allowed in output buffer.')
        if self.disconnecting:
            return
        self.outQueue.append(data)
        if sum(map(len, self.outQueue)) > FULL_BUFFER_SIZE:
            self.bufferFull()

    def checkWork(self):
        numBytesWritten = 0
        if not self.outQueue:
            if self.disconnecting:
                self.writeConnectionLost()
                return 0
            try:
                win32file.WriteFile(self.writePipe, b'', None)
            except pywintypes.error:
                self.writeConnectionLost()
                return numBytesWritten
        while self.outQueue:
            data = self.outQueue.pop(0)
            errCode = 0
            try:
                errCode, nBytesWritten = win32file.WriteFile(self.writePipe, data, None)
            except win32api.error:
                self.writeConnectionLost()
                break
            else:
                numBytesWritten += nBytesWritten
                if len(data) > nBytesWritten:
                    self.outQueue.insert(0, data[nBytesWritten:])
                    break
        else:
            resumed = self.bufferEmpty()
            if not resumed and self.disconnecting:
                self.writeConnectionLost()
        return numBytesWritten