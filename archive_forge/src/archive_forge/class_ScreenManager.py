from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
class ScreenManager(FloatLayout):
    """Screen manager. This is the main class that will control your
    :class:`Screen` stack and memory.

    By default, the manager will show only one screen at a time.
    """
    current = StringProperty(None, allownone=True)
    "\n    Name of the screen currently shown, or the screen to show.\n\n    ::\n\n        from kivy.uix.screenmanager import ScreenManager, Screen\n\n        sm = ScreenManager()\n        sm.add_widget(Screen(name='first'))\n        sm.add_widget(Screen(name='second'))\n\n        # By default, the first added screen will be shown. If you want to\n        # show another one, just set the 'current' property.\n        sm.current = 'second'\n\n    :attr:`current` is a :class:`~kivy.properties.StringProperty` and defaults\n    to None.\n    "
    transition = ObjectProperty(baseclass=TransitionBase)
    "Transition object to use for animating the transition from the current\n    screen to the next one being shown.\n\n    For example, if you want to use a :class:`WipeTransition` between\n    slides::\n\n        from kivy.uix.screenmanager import ScreenManager, Screen,\n        WipeTransition\n\n        sm = ScreenManager(transition=WipeTransition())\n        sm.add_widget(Screen(name='first'))\n        sm.add_widget(Screen(name='second'))\n\n        # by default, the first added screen will be shown. If you want to\n        # show another one, just set the 'current' property.\n        sm.current = 'second'\n\n    :attr:`transition` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to a :class:`SlideTransition`.\n\n    .. versionchanged:: 1.8.0\n\n        Default transition has been changed from :class:`SwapTransition` to\n        :class:`SlideTransition`.\n    "
    screens = ListProperty()
    'List of all the :class:`Screen` widgets added. You should not change\n    this list manually. Use the\n    :meth:`add_widget <kivy.uix.widget.Widget.add_widget>` method instead.\n\n    :attr:`screens` is a :class:`~kivy.properties.ListProperty` and defaults to\n    [], read-only.\n    '
    current_screen = ObjectProperty(None, allownone=True)
    'Contains the currently displayed screen. You must not change this\n    property manually, use :attr:`current` instead.\n\n    :attr:`current_screen` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None, read-only.\n    '

    def _get_screen_names(self):
        return [s.name for s in self.screens]
    screen_names = AliasProperty(_get_screen_names, bind=('screens',))
    'List of the names of all the :class:`Screen` widgets added. The list\n    is read only.\n\n    :attr:`screens_names` is an :class:`~kivy.properties.AliasProperty` and\n    is read-only. It is updated if the screen list changes or the name\n    of a screen changes.\n    '

    def __init__(self, **kwargs):
        if 'transition' not in kwargs:
            self.transition = SlideTransition()
        super(ScreenManager, self).__init__(**kwargs)
        self.fbind('pos', self._update_pos)

    def _screen_name_changed(self, screen, name):
        self.property('screen_names').dispatch(self)
        if screen == self.current_screen:
            self.current = name

    def add_widget(self, widget, *args, **kwargs):
        """
        .. versionchanged:: 2.1.0
            Renamed argument `screen` to `widget`.
        """
        if not isinstance(widget, Screen):
            raise ScreenManagerException('ScreenManager accepts only Screen widget.')
        if widget.manager:
            if widget.manager is self:
                raise ScreenManagerException('Screen already managed by this ScreenManager (are you calling `switch_to` when you should be setting `current`?)')
            raise ScreenManagerException('Screen already managed by another ScreenManager.')
        widget.manager = self
        widget.bind(name=self._screen_name_changed)
        self.screens.append(widget)
        if self.current is None:
            self.current = widget.name

    def remove_widget(self, widget, *args, **kwargs):
        if not isinstance(widget, Screen):
            raise ScreenManagerException('ScreenManager uses remove_widget only for removing Screens.')
        if widget not in self.screens:
            return
        if self.current_screen == widget:
            other = next(self)
            if widget.name == other:
                self.current = None
                widget.parent.real_remove_widget(widget)
            else:
                self.current = other
        widget.manager = None
        widget.unbind(name=self._screen_name_changed)
        self.screens.remove(widget)

    def clear_widgets(self, children=None, *args, **kwargs):
        """
        .. versionchanged:: 2.1.0
            Renamed argument `screens` to `children`.
        """
        if children is None:
            children = self.screens[:]
        remove_widget = self.remove_widget
        for widget in children:
            remove_widget(widget)

    def real_add_widget(self, screen, *args):
        parent = screen.parent
        if parent:
            parent.real_remove_widget(screen)
        super(ScreenManager, self).add_widget(screen)

    def real_remove_widget(self, screen, *args):
        super(ScreenManager, self).remove_widget(screen)

    def on_current(self, instance, value):
        if value is None:
            self.transition.stop()
            self.current_screen = None
            return
        screen = self.get_screen(value)
        if screen == self.current_screen:
            return
        self.transition.stop()
        previous_screen = self.current_screen
        self.current_screen = screen
        if previous_screen:
            self.transition.screen_in = screen
            self.transition.screen_out = previous_screen
            self.transition.start(self)
        else:
            self.real_add_widget(screen)
            screen.pos = self.pos
            self.do_layout()
            screen.dispatch('on_pre_enter')
            screen.dispatch('on_enter')

    def get_screen(self, name):
        """Return the screen widget associated with the name or raise a
        :class:`ScreenManagerException` if not found.
        """
        matches = [s for s in self.screens if s.name == name]
        num_matches = len(matches)
        if num_matches == 0:
            raise ScreenManagerException('No Screen with name "%s".' % name)
        if num_matches > 1:
            Logger.warn('Multiple screens named "%s": %s' % (name, matches))
        return matches[0]

    def has_screen(self, name):
        """Return True if a screen with the `name` has been found.

        .. versionadded:: 1.6.0
        """
        return bool([s for s in self.screens if s.name == name])

    def __next__(self):
        """Py2K backwards compatibility without six or other lib.
        """
        screens = self.screens
        if not screens:
            return
        try:
            index = screens.index(self.current_screen)
            index = (index + 1) % len(screens)
            return screens[index].name
        except ValueError:
            return

    def next(self):
        """Return the name of the next screen from the screen list."""
        return self.__next__()

    def previous(self):
        """Return the name of the previous screen from the screen list.
        """
        screens = self.screens
        if not screens:
            return
        try:
            index = screens.index(self.current_screen)
            index = (index - 1) % len(screens)
            return screens[index].name
        except ValueError:
            return

    def switch_to(self, screen, **options):
        """Add a new or existing screen to the ScreenManager and switch to it.
        The previous screen will be "switched away" from. `options` are the
        :attr:`transition` options that will be changed before the animation
        happens.

        If no previous screens are available, the screen will be used as the
        main one::

            sm = ScreenManager()
            sm.switch_to(screen1)
            # later
            sm.switch_to(screen2, direction='left')
            # later
            sm.switch_to(screen3, direction='right', duration=1.)

        If any animation is in progress, it will be stopped and replaced by
        this one: you should avoid this because the animation will just look
        weird. Use either :meth:`switch_to` or :attr:`current` but not both.

        The `screen` name will be changed if there is any conflict with the
        current screen.

        .. versionadded: 1.8.0
        """
        assert screen is not None
        if not isinstance(screen, Screen):
            raise ScreenManagerException('ScreenManager accepts only Screen widget.')
        self.transition.stop()
        if screen not in self.screens:
            if self.has_screen(screen.name):
                screen.name = self._generate_screen_name()
        old_transition = self.transition
        specified_transition = options.pop('transition', None)
        if specified_transition:
            self.transition = specified_transition
        for key, value in iteritems(options):
            setattr(self.transition, key, value)
        if screen.manager is not self:
            self.add_widget(screen)
        if self.current_screen is screen:
            return
        old_current = self.current_screen

        def remove_old_screen(transition):
            if old_current in self.children:
                self.remove_widget(old_current)
                self.transition = old_transition
            transition.unbind(on_complete=remove_old_screen)
        self.transition.bind(on_complete=remove_old_screen)
        self.current = screen.name

    def _generate_screen_name(self):
        i = 0
        while True:
            name = '_screen{}'.format(i)
            if not self.has_screen(name):
                return name
            i += 1

    def _update_pos(self, instance, value):
        for child in self.children:
            if self.transition.is_active and (child == self.transition.screen_in or child == self.transition.screen_out):
                continue
            child.pos = value

    def on_motion(self, etype, me):
        if self.transition.is_active:
            return False
        return super().on_motion(etype, me)

    def on_touch_down(self, touch):
        if self.transition.is_active:
            return False
        return super(ScreenManager, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.transition.is_active:
            return False
        return super(ScreenManager, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.transition.is_active:
            return False
        return super(ScreenManager, self).on_touch_up(touch)