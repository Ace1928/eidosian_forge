import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def _bind_button_control(self, relation, control, button_name):
    if button_name in ('dpleft', 'dpright', 'dpup', 'dpdown'):

        @control.event
        def on_change(value):
            setattr(self, button_name, value)
            self.dispatch_event('on_dpad_motion', self, self.dpleft, self.dpright, self.dpup, self.dpdown)
    else:

        @control.event
        def on_change(value):
            setattr(self, button_name, value)

        @control.event
        def on_press():
            self.dispatch_event('on_button_press', self, button_name)

        @control.event
        def on_release():
            self.dispatch_event('on_button_release', self, button_name)