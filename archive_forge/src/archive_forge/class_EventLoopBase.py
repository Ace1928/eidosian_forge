import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
class EventLoopBase(EventDispatcher):
    """Main event loop. This loop handles the updating of input and
    dispatching events.
    """
    __events__ = ('on_start', 'on_pause', 'on_stop')

    def __init__(self):
        super(EventLoopBase, self).__init__()
        self.quit = False
        self.input_events = []
        self.postproc_modules = []
        self.status = 'idle'
        self.stopping = False
        self.input_providers = []
        self.input_providers_autoremove = []
        self.event_listeners = []
        self.window = None
        self.me_list = []

    @property
    def touches(self):
        """Return the list of all touches currently in down or move states.
        """
        return self.me_list

    def ensure_window(self):
        """Ensure that we have a window.
        """
        import kivy.core.window
        if not self.window:
            Logger.critical('App: Unable to get a Window, abort.')
            sys.exit(1)

    def set_window(self, window):
        """Set the window used for the event loop.
        """
        self.window = window

    def add_input_provider(self, provider, auto_remove=False):
        """Add a new input provider to listen for touch events.
        """
        if provider not in self.input_providers:
            self.input_providers.append(provider)
            if auto_remove:
                self.input_providers_autoremove.append(provider)

    def remove_input_provider(self, provider):
        """Remove an input provider.

        .. versionchanged:: 2.1.0
            Provider will be also removed if it exist in auto-remove list.
        """
        if provider in self.input_providers:
            self.input_providers.remove(provider)
            if provider in self.input_providers_autoremove:
                self.input_providers_autoremove.remove(provider)

    def add_event_listener(self, listener):
        """Add a new event listener for getting touch events.
        """
        if listener not in self.event_listeners:
            self.event_listeners.append(listener)

    def remove_event_listener(self, listener):
        """Remove an event listener from the list.
        """
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)

    def start(self):
        """Must be called before :meth:`EventLoopBase.run()`. This starts all
        configured input providers.

        .. versionchanged:: 2.1.0
            Method can be called multiple times, but event loop will start only
            once.
        """
        if self.status == 'started':
            return
        self.status = 'started'
        self.quit = False
        Clock.start_clock()
        for provider in self.input_providers:
            provider.start()
        self.dispatch('on_start')

    def close(self):
        """Exit from the main loop and stop all configured
        input providers."""
        self.quit = True
        self.stop()
        self.status = 'closed'

    def stop(self):
        """Stop all input providers and call callbacks registered using
        `EventLoop.add_stop_callback()`.

        .. versionchanged:: 2.1.0
            Method can be called multiple times, but event loop will stop only
            once.
        """
        if self.status != 'started':
            return
        for provider in reversed(self.input_providers[:]):
            provider.stop()
            self.remove_input_provider(provider)
        self.input_events = []
        Clock.stop_clock()
        self.stopping = False
        self.status = 'stopped'
        self.dispatch('on_stop')

    def add_postproc_module(self, mod):
        """Add a postproc input module (DoubleTap, TripleTap, DeJitter
        RetainTouch are defaults)."""
        if mod not in self.postproc_modules:
            self.postproc_modules.append(mod)

    def remove_postproc_module(self, mod):
        """Remove a postproc module."""
        if mod in self.postproc_modules:
            self.postproc_modules.remove(mod)

    def remove_android_splash(self, *args):
        """Remove android presplash in SDL2 bootstrap."""
        try:
            from android import remove_presplash
            remove_presplash()
        except ImportError:
            Logger.warning('Base: Failed to import "android" module. Could not remove android presplash.')
            return

    def post_dispatch_input(self, etype, me):
        """This function is called by :meth:`EventLoopBase.dispatch_input()`
        when we want to dispatch an input event. The event is dispatched to
        all listeners and if grabbed, it's dispatched to grabbed widgets.
        """
        if etype == 'begin':
            self.me_list.append(me)
        elif etype == 'end':
            if me in self.me_list:
                self.me_list.remove(me)
        if not me.grab_exclusive_class:
            for listener in self.event_listeners:
                listener.dispatch('on_motion', etype, me)
        if not me.is_touch:
            return
        me.grab_state = True
        for weak_widget in me.grab_list[:]:
            wid = weak_widget()
            if wid is None:
                me.grab_list.remove(weak_widget)
                continue
            root_window = wid.get_root_window()
            if wid != root_window and root_window is not None:
                me.push()
                try:
                    root_window.transform_motion_event_2d(me, wid)
                except AttributeError:
                    me.pop()
                    continue
            me.grab_current = wid
            wid._context.push()
            if etype == 'begin':
                pass
            elif etype == 'update':
                if wid._context.sandbox:
                    with wid._context.sandbox:
                        wid.dispatch('on_touch_move', me)
                else:
                    wid.dispatch('on_touch_move', me)
            elif etype == 'end':
                if wid._context.sandbox:
                    with wid._context.sandbox:
                        wid.dispatch('on_touch_up', me)
                else:
                    wid.dispatch('on_touch_up', me)
            wid._context.pop()
            me.grab_current = None
            if wid != root_window and root_window is not None:
                me.pop()
        me.grab_state = False
        me.dispatch_done()

    def _dispatch_input(self, *ev):
        if ev in self.input_events:
            self.input_events.remove(ev)
        self.input_events.append(ev)

    def dispatch_input(self):
        """Called by :meth:`EventLoopBase.idle()` to read events from input
        providers, pass events to postproc, and dispatch final events.
        """
        for provider in self.input_providers:
            provider.update(dispatch_fn=self._dispatch_input)
        for mod in self.postproc_modules:
            self.input_events = mod.process(events=self.input_events)
        input_events = self.input_events
        pop = input_events.pop
        post_dispatch_input = self.post_dispatch_input
        while input_events:
            post_dispatch_input(*pop(0))

    def mainloop(self):
        while not self.quit and self.status == 'started':
            try:
                self.idle()
                if self.window:
                    self.window.mainloop()
            except BaseException as inst:
                r = ExceptionManager.handle_exception(inst)
                if r == ExceptionManager.RAISE:
                    stopTouchApp()
                    raise
                else:
                    pass

    async def async_mainloop(self):
        while not self.quit and self.status == 'started':
            try:
                await self.async_idle()
                if self.window:
                    self.window.mainloop()
            except BaseException as inst:
                r = ExceptionManager.handle_exception(inst)
                if r == ExceptionManager.RAISE:
                    stopTouchApp()
                    raise
                else:
                    pass
        Logger.info('Window: exiting mainloop and closing.')
        self.close()

    def idle(self):
        """This function is called after every frame. By default:

           * it "ticks" the clock to the next frame.
           * it reads all input and dispatches events.
           * it dispatches `on_update`, `on_draw` and `on_flip` events to the
             window.
        """
        Clock.tick()
        if not self.quit:
            self.dispatch_input()
        if not self.quit:
            Builder.sync()
        if not self.quit:
            Clock.tick_draw()
        if not self.quit:
            Builder.sync()
        if not self.quit:
            window = self.window
            if window and window.canvas.needs_redraw:
                window.dispatch('on_draw')
                window.dispatch('on_flip')
        if len(self.event_listeners) == 0:
            Logger.error('Base: No event listeners have been created')
            Logger.error('Base: Application will leave')
            self.exit()
            return False
        return self.quit

    async def async_idle(self):
        """Identical to :meth:`idle`, but instead used when running
        within an async event loop.
        """
        await Clock.async_tick()
        if not self.quit:
            self.dispatch_input()
        if not self.quit:
            Builder.sync()
        if not self.quit:
            Clock.tick_draw()
        if not self.quit:
            Builder.sync()
        if not self.quit:
            window = self.window
            if window and window.canvas.needs_redraw:
                window.dispatch('on_draw')
                window.dispatch('on_flip')
        if len(self.event_listeners) == 0:
            Logger.error('Base: No event listeners have been created')
            Logger.error('Base: Application will leave')
            self.exit()
            return False
        return self.quit

    def run(self):
        """Main loop"""
        while not self.quit:
            self.idle()
        self.exit()

    def exit(self):
        """Close the main loop and close the window."""
        self.close()
        if self.window:
            self.window.close()

    def on_stop(self):
        """Event handler for `on_stop` events which will be fired right
        after all input providers have been stopped."""
        pass

    def on_pause(self):
        """Event handler for `on_pause` which will be fired when
        the event loop is paused."""
        pass

    def on_start(self):
        """Event handler for `on_start` which will be fired right
        after all input providers have been started."""
        pass