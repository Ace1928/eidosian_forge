from time import time
from kivy.event import EventDispatcher
from kivy.properties import NumericProperty, BooleanProperty
from kivy.clock import Clock
class KineticEffect(EventDispatcher):
    """Kinetic effect class. See module documentation for more information.
    """
    velocity = NumericProperty(0)
    'Velocity of the movement.\n\n    :attr:`velocity` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    friction = NumericProperty(0.05)
    'Friction to apply on the velocity\n\n    :attr:`friction` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.05.\n    '
    value = NumericProperty(0)
    'Value (during the movement and computed) of the effect.\n\n    :attr:`value` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    is_manual = BooleanProperty(False)
    'Indicate if a movement is in progress (True) or not (False).\n\n    :attr:`is_manual` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    max_history = NumericProperty(5)
    'Save up to `max_history` movement value into the history. This is used\n    for correctly calculating the velocity according to the movement.\n\n    :attr:`max_history` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 5.\n    '
    min_distance = NumericProperty(0.1)
    'The minimal distance for a movement to have nonzero velocity.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`min_distance` is :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.1.\n    '
    min_velocity = NumericProperty(0.5)
    'Velocity below this quantity is normalized to 0. In other words,\n    any motion whose velocity falls below this number is stopped.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`min_velocity` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.5.\n    '
    std_dt = NumericProperty(0.017)
    ' std_dt\n        correction update_velocity if dt is not constant\n\n    .. versionadded:: 2.0.0\n\n    :attr:`std_dt` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.017.\n    '

    def __init__(self, **kwargs):
        self.history = []
        self.trigger_velocity_update = Clock.create_trigger(self.update_velocity, 0)
        super(KineticEffect, self).__init__(**kwargs)

    def apply_distance(self, distance):
        if abs(distance) < self.min_distance:
            self.velocity = 0
        self.value += distance

    def start(self, val, t=None):
        """Start the movement.

        :Parameters:
            `val`: float or int
                Value of the movement
            `t`: float, defaults to None
                Time when the movement happen. If no time is set, it will use
                time.time()
        """
        self.is_manual = True
        t = t or time()
        self.velocity = 0
        self.history = [(t, val)]

    def update(self, val, t=None):
        """Update the movement.

        See :meth:`start` for the arguments.
        """
        t = t or time()
        distance = val - self.history[-1][1]
        self.apply_distance(distance)
        self.history.append((t, val))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def stop(self, val, t=None):
        """Stop the movement.

        See :meth:`start` for the arguments.
        """
        self.is_manual = False
        t = t or time()
        distance = val - self.history[-1][1]
        self.apply_distance(distance)
        newest_sample = (t, val)
        old_sample = self.history[0]
        for sample in self.history:
            if newest_sample[0] - sample[0] < 10.0 / 60.0:
                break
            old_sample = sample
        distance = newest_sample[1] - old_sample[1]
        duration = abs(newest_sample[0] - old_sample[0])
        self.velocity = distance / max(duration, 0.0001)
        self.trigger_velocity_update()

    def cancel(self):
        """Cancel a movement. This can be used in case :meth:`stop` cannot be
        called. It will reset :attr:`is_manual` to False, and compute the
        movement if the velocity is > 0.
        """
        self.is_manual = False
        self.trigger_velocity_update()

    def update_velocity(self, dt):
        """(internal) Update the velocity according to the frametime and
        friction.
        """
        if abs(self.velocity) <= self.min_velocity:
            self.velocity = 0
            return
        self.velocity -= self.velocity * self.friction * dt / self.std_dt
        self.apply_distance(self.velocity * dt)
        self.trigger_velocity_update()