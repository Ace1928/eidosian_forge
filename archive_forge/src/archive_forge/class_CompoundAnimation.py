from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
class CompoundAnimation(Animation):

    def stop_property(self, widget, prop):
        self.anim1.stop_property(widget, prop)
        self.anim2.stop_property(widget, prop)
        if not self.anim1.have_properties_to_animate(widget) and (not self.anim2.have_properties_to_animate(widget)):
            self.stop(widget)

    def cancel(self, widget):
        self.anim1.cancel(widget)
        self.anim2.cancel(widget)
        super().cancel(widget)

    def cancel_property(self, widget, prop):
        """Even if an animation is running, remove a property. It will not be
        animated further. If it was the only/last property being animated,
        the animation will be canceled (see :attr:`cancel`)

        This method overrides `:class:kivy.animation.Animation`'s
        version, to cancel it on all animations of the Sequence.

        .. versionadded:: 1.10.0
        """
        self.anim1.cancel_property(widget, prop)
        self.anim2.cancel_property(widget, prop)
        if not self.anim1.have_properties_to_animate(widget) and (not self.anim2.have_properties_to_animate(widget)):
            self.cancel(widget)

    def have_properties_to_animate(self, widget):
        return self.anim1.have_properties_to_animate(widget) or self.anim2.have_properties_to_animate(widget)

    @property
    def animated_properties(self):
        return ChainMap({}, self.anim2.animated_properties, self.anim1.animated_properties)

    @property
    def transition(self):
        raise AttributeError("Can't lookup transition attribute of a CompoundAnimation")