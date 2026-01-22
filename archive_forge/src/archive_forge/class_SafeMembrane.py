import inspect
from threading import Thread, Event
from kivy.app import App
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.utils import deprecated
class SafeMembrane(object):
    """
    This help is for a proxy object. Did you want help on the proxy's referent
    instead? Try using help(<instance>._ref)

    The SafeMembrane is a threadsafe proxy that also returns attributes as new
    thread-safe objects
    and makes thread-safe method calls, preventing thread-unsafe objects
    from leaking into the user's environment.
    """
    __slots__ = ('_ref', 'safe', 'confirmed')

    def __init__(self, ob, *args, **kwargs):
        self.confirmed = EventLoop.confirmed
        self.safe = EventLoop.safe
        self._ref = ob

    def safeIn(self):
        """Provides a thread-safe entry point for interactive launching."""
        self.safe.clear()
        Clock.schedule_once(safeWait, -1)
        self.confirmed.wait()

    def safeOut(self):
        """Provides a thread-safe exit point for interactive launching."""
        self.safe.set()

    def isMethod(self, fn):
        return inspect.ismethod(fn)

    def __repr__(self):
        return self._ref.__repr__()

    def __call__(self, *args, **kw):
        self.safeIn()
        args = list(map(unwrap, args))
        for k in list(kw.keys()):
            kw[k] = unwrap(kw[k])
        r = self._ref(*args, **kw)
        self.safeOut()
        if r is not None:
            return SafeMembrane(r)

    def __getattribute__(self, attr, oga=object.__getattribute__):
        if attr.startswith('__') or attr == '_ref':
            subject = oga(self, '_ref')
            if attr == '_ref':
                return subject
            return getattr(subject, attr)
        return oga(self, attr)

    def __getattr__(self, attr, oga=object.__getattribute__):
        r = getattr(oga(self, '_ref'), attr)
        return SafeMembrane(r)

    def __setattr__(self, attr, val, osa=object.__setattr__):
        if attr == '_ref' or (hasattr(type(self), attr) and (not attr.startswith('__'))):
            osa(self, attr, val)
        else:
            self.safeIn()
            val = unwrap(val)
            setattr(self._ref, attr, val)
            self.safeOut()

    def __delattr__(self, attr, oda=object.__delattr__):
        self.safeIn()
        delattr(self._ref, attr)
        self.safeOut()

    def __bool__(self):
        return bool(self._ref)

    def __getitem__(self, arg):
        return SafeMembrane(self._ref[arg])

    def __setitem__(self, arg, val):
        self.safeIn()
        val = unwrap(val)
        self._ref[arg] = val
        self.safeOut()

    def __delitem__(self, arg):
        self.safeIn()
        del self._ref[arg]
        self.safeOut()

    def __getslice__(self, i, j):
        return SafeMembrane(self._ref[i:j])

    def __setslice__(self, i, j, val):
        self.safeIn()
        val = unwrap(val)
        self._ref[i:j] = val
        self.safeOut()

    def __delslice__(self, i, j):
        self.safeIn()
        del self._ref[i:j]
        self.safeOut()

    def __enter__(self, *args, **kwargs):
        self.safeIn()
        self._ref.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self._ref.__exit__(*args, **kwargs)
        self.safeOut()