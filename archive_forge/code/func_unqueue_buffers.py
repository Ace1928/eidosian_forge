import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def unqueue_buffers(self):
    processed = self.buffers_processed
    assert _debug('Processed buffer count: {}'.format(processed))
    if processed > 0:
        buffers = (al.ALuint * processed)()
        al.alSourceUnqueueBuffers(self._al_source, len(buffers), buffers)
        self._check_error('Failed to unqueue buffers from source.')
        self.buffer_pool.return_buffers([self._owned_buffers.pop(bn) for bn in buffers])
    return processed