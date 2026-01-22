from functools import wraps
from kivy.context import Context
from kivy.base import ExceptionManagerBase
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
class Sandbox(FloatLayout):
    """Sandbox widget, used to trap all the exceptions raised by child
    widgets.
    """

    def __init__(self, **kwargs):
        self._context = Context(init=True)
        self._context['ExceptionManager'] = SandboxExceptionManager(self)
        self._context.sandbox = self
        self._context.push()
        self.on_context_created()
        self._container = None
        super(Sandbox, self).__init__(**kwargs)
        self._container = SandboxContent(size=self.size, pos=self.pos)
        super(Sandbox, self).add_widget(self._container)
        self._context.pop()
        Clock.schedule_interval(self._clock_sandbox, 0)
        Clock.schedule_once(self._clock_sandbox_draw, -1)
        self.main_clock = object.__getattribute__(Clock, '_obj')

    def __enter__(self):
        self._context.push()

    def __exit__(self, _type, value, traceback):
        self._context.pop()
        if _type is not None:
            return self.on_exception(value, _traceback=traceback)

    def on_context_created(self):
        """Override this method in order to load your kv file or do anything
        else with the newly created context.
        """
        pass

    def on_exception(self, exception, _traceback=None):
        """Override this method in order to catch all the exceptions from
        children.

        If you return True, it will not reraise the exception.
        If you return False, the exception will be raised to the parent.
        """
        import traceback
        traceback.print_tb(_traceback)
        return True
    on_motion = sandbox(Widget.on_motion)
    on_touch_down = sandbox(Widget.on_touch_down)
    on_touch_move = sandbox(Widget.on_touch_move)
    on_touch_up = sandbox(Widget.on_touch_up)

    @sandbox
    def add_widget(self, *args, **kwargs):
        self._container.add_widget(*args, **kwargs)

    @sandbox
    def remove_widget(self, *args, **kwargs):
        self._container.remove_widget(*args, **kwargs)

    @sandbox
    def clear_widgets(self, *args, **kwargs):
        self._container.clear_widgets(*args, **kwargs)

    @sandbox
    def on_size(self, *args):
        if self._container:
            self._container.size = self.size

    @sandbox
    def on_pos(self, *args):
        if self._container:
            self._container.pos = self.pos

    @sandbox
    def _clock_sandbox(self, dt):
        Clock.tick()
        Builder.sync()

    @sandbox
    def _clock_sandbox_draw(self, dt):
        Clock.tick_draw()
        Builder.sync()
        self.main_clock.schedule_once(self._call_draw, 0)

    def _call_draw(self, dt):
        self.main_clock.schedule_once(self._clock_sandbox_draw, -1)