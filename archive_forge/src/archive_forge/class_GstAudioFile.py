import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
class GstAudioFile(AudioFile):
    """Reads raw audio data from any audio file that Gstreamer
    knows how to decode.

        >>> with GstAudioFile('something.mp3') as f:
        >>>     print f.samplerate
        >>>     print f.channels
        >>>     print f.duration
        >>>     for block in f:
        >>>         do_something(block)

    Iterating the object yields blocks of 16-bit PCM data. Three
    pieces of stream information are also available: samplerate (in Hz),
    number of channels, and duration (in seconds).

    It's very important that the client call close() when it's done
    with the object. Otherwise, the program is likely to hang on exit.
    Alternatively, of course, one can just use the file as a context
    manager, as shown above.
    """

    def __init__(self, path):
        self.running = False
        self.finished = False
        self.pipeline = Gst.Pipeline()
        self.dec = Gst.ElementFactory.make('uridecodebin', None)
        self.conv = Gst.ElementFactory.make('audioconvert', None)
        self.sink = Gst.ElementFactory.make('appsink', None)
        if self.dec is None or self.conv is None or self.sink is None:
            raise IncompleteGStreamerError()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::eos', self._message)
        bus.connect('message::error', self._message)
        uri = 'file://' + quote(os.path.abspath(path))
        self.dec.set_property('uri', uri)
        self.dec.connect('pad-added', self._pad_added)
        self.dec.connect('no-more-pads', self._no_more_pads)
        self.dec.connect('unknown-type', self._unkown_type)
        self.sink.set_property('caps', Gst.Caps.from_string('audio/x-raw, format=(string)S16LE'))
        self.sink.set_property('drop', False)
        self.sink.set_property('max-buffers', BUFFER_SIZE)
        self.sink.set_property('sync', False)
        self.sink.set_property('emit-signals', True)
        self.sink.connect('new-sample', self._new_sample)
        self.ready_sem = threading.Semaphore(0)
        self.caps_handler = self.sink.get_static_pad('sink').connect('notify::caps', self._notify_caps)
        self.pipeline.add(self.dec)
        self.pipeline.add(self.conv)
        self.pipeline.add(self.sink)
        self.conv.link(self.sink)
        self.queue = queue.Queue(QUEUE_SIZE)
        self.thread = get_loop_thread()
        self.read_exc = None
        self.running = True
        self.got_caps = False
        self.pipeline.set_state(Gst.State.PLAYING)
        self.ready_sem.acquire()
        if self.read_exc:
            self.close(True)
            raise self.read_exc

    def _notify_caps(self, pad, args):
        """The callback for the sinkpad's "notify::caps" signal.
        """
        self.got_caps = True
        info = pad.get_current_caps().get_structure(0)
        self.channels = info.get_int('channels')[1]
        self.samplerate = info.get_int('rate')[1]
        success, length = pad.get_peer().query_duration(Gst.Format.TIME)
        if success:
            self.duration = length / 1000000000
        else:
            self.read_exc = MetadataMissingError('duration not available')
        self.ready_sem.release()
    _got_a_pad = False

    def _pad_added(self, element, pad):
        """The callback for GstElement's "pad-added" signal.
        """
        name = pad.query_caps(None).to_string()
        if name.startswith('audio/x-raw'):
            nextpad = self.conv.get_static_pad('sink')
            if not nextpad.is_linked():
                self._got_a_pad = True
                pad.link(nextpad)

    def _no_more_pads(self, element):
        """The callback for GstElement's "no-more-pads" signal.
        """
        if not self._got_a_pad:
            self.read_exc = NoStreamError()
            self.ready_sem.release()

    def _new_sample(self, sink):
        """The callback for appsink's "new-sample" signal.
        """
        if self.running:
            buf = sink.emit('pull-sample').get_buffer()
            mem = buf.get_all_memory()
            success, info = mem.map(Gst.MapFlags.READ)
            if success:
                if isinstance(info.data, memoryview):
                    data = bytes(info.data)
                else:
                    data = info.data
                mem.unmap(info)
                self.queue.put(data)
            else:
                raise GStreamerError('Unable to map buffer memory while reading the file.')
        return Gst.FlowReturn.OK

    def _unkown_type(self, uridecodebin, decodebin, caps):
        """The callback for decodebin's "unknown-type" signal.
        """
        streaminfo = caps.to_string()
        if not streaminfo.startswith('audio/'):
            return
        self.read_exc = UnknownTypeError(streaminfo)
        self.ready_sem.release()

    def _message(self, bus, message):
        """The callback for GstBus's "message" signal (for two kinds of
        messages).
        """
        if not self.finished:
            if message.type == Gst.MessageType.EOS:
                self.queue.put(SENTINEL)
                if not self.got_caps:
                    self.read_exc = NoStreamError()
                    self.ready_sem.release()
            elif message.type == Gst.MessageType.ERROR:
                gerror, debug = message.parse_error()
                if 'not-linked' in debug:
                    self.read_exc = NoStreamError()
                elif 'No such file' in debug:
                    self.read_exc = IOError('resource not found')
                else:
                    self.read_exc = FileReadError(debug)
                self.ready_sem.release()

    def __next__(self):
        val = self.queue.get()
        if val == SENTINEL:
            raise StopIteration
        return val

    def __iter__(self):
        return self

    def close(self, force=False):
        """Close the file and clean up associated resources.

        Calling `close()` a second time has no effect.
        """
        if self.running or force:
            self.running = False
            self.finished = True
            self.pipeline.get_bus().remove_signal_watch()
            self.dec.set_property('uri', None)
            self.sink.get_static_pad('sink').disconnect(self.caps_handler)
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.pipeline.set_state(Gst.State.NULL)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False