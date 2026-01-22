import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def _runTouchApp_prepare(widget=None):
    from kivy.input import MotionEventFactory, kivy_postproc_modules
    if widget:
        EventLoop.ensure_window()
    for key, value in Config.items('input'):
        Logger.debug('Base: Create provider from %s' % str(value))
        args = str(value).split(',', 1)
        if len(args) == 1:
            args.append('')
        provider_id, args = args
        provider = MotionEventFactory.get(provider_id)
        if provider is None:
            Logger.warning('Base: Unknown <%s> provider' % str(provider_id))
            continue
        p = provider(key, args)
        if p:
            EventLoop.add_input_provider(p, True)
    for mod in list(kivy_postproc_modules.values()):
        EventLoop.add_postproc_module(mod)
    if widget and EventLoop.window:
        if widget not in EventLoop.window.children:
            EventLoop.window.add_widget(widget)
    Logger.info('Base: Start application main loop')
    EventLoop.start()
    if platform == 'android':
        Clock.schedule_once(EventLoop.remove_android_splash)