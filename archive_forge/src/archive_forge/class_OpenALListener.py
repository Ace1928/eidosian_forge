import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALListener(OpenALObject):

    @property
    def position(self):
        return self._get_3floats(al.AL_POSITION)

    @position.setter
    def position(self, values):
        self._set_3floats(al.AL_POSITION, values)

    @property
    def velocity(self):
        return self._get_3floats(al.AL_VELOCITY)

    @velocity.setter
    def velocity(self, values):
        self._set_3floats(al.AL_VELOCITY, values)

    @property
    def gain(self):
        return self._get_float(al.AL_GAIN)

    @gain.setter
    def gain(self, value):
        self._set_float(al.AL_GAIN, value)

    @property
    def orientation(self):
        values = self._get_float_vector(al.AL_ORIENTATION, 6)
        return OpenALOrientation(values[0:3], values[3:6])

    @orientation.setter
    def orientation(self, values):
        if len(values) == 2:
            actual_values = values[0] + values[1]
        elif len(values) == 6:
            actual_values = values
        else:
            actual_values = []
        if len(actual_values) != 6:
            raise ValueError('Need 2 tuples of 3 or 1 tuple of 6.')
        self._set_float_vector(al.AL_ORIENTATION, actual_values)

    def _get_float(self, key):
        al_float = al.ALfloat()
        al.alGetListenerf(key, al_float)
        self._check_error('Failed to get value')
        return al_float.value

    def _set_float(self, key, value):
        al.alListenerf(key, float(value))
        self._check_error('Failed to set value.')

    def _get_3floats(self, key):
        x = al.ALfloat()
        y = al.ALfloat()
        z = al.ALfloat()
        al.alGetListener3f(key, x, y, z)
        self._check_error('Failed to get value')
        return (x.value, y.value, z.value)

    def _set_3floats(self, key, values):
        x, y, z = map(float, values)
        al.alListener3f(key, x, y, z)
        self._check_error('Failed to set value.')

    def _get_float_vector(self, key, count):
        al_float_vector = (al.ALfloat * count)()
        al.alGetListenerfv(key, al_float_vector)
        self._check_error('Failed to get value')
        return [x for x in al_float_vector]

    def _set_float_vector(self, key, values):
        al_float_vector = (al.ALfloat * len(values))(*values)
        al.alListenerfv(key, al_float_vector)
        self._check_error('Failed to set value.')