from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class TuioMotionEventProvider(MotionEventProvider):
    """The TUIO provider listens to a socket and handles some of the incoming
    OSC messages:

        * /tuio/2Dcur
        * /tuio/2Dobj

    You can easily extend the provider to handle new TUIO paths like so::

        # Create a class to handle the new TUIO type/path
        # Replace NEWPATH with the pathname you want to handle
        class TuioNEWPATHMotionEvent(MotionEvent):

            def depack(self, args):
                # In this method, implement 'unpacking' for the received
                # arguments. you basically translate from TUIO args to Kivy
                # MotionEvent variables. If all you receive are x and y
                # values, you can do it like this:
                if len(args) == 2:
                    self.sx, self.sy = args
                    self.profile = ('pos', )
                self.sy = 1 - self.sy
                super().depack(args)

        # Register it with the TUIO MotionEvent provider.
        # You obviously need to replace the PATH placeholders appropriately.
        TuioMotionEventProvider.register('/tuio/PATH', TuioNEWPATHMotionEvent)

    .. note::

        The class name is of no technical importance. Your class will be
        associated with the path that you pass to the ``register()``
        function. To keep things simple, you should name your class after the
        path that it handles, though.
    """
    __handlers__ = {}

    def __init__(self, device, args):
        super().__init__(device, args)
        args = args.split(',')
        if len(args) == 0:
            Logger.error('Tuio: Invalid configuration for TUIO provider')
            Logger.error('Tuio: Format must be ip:port (eg. 127.0.0.1:3333)')
            err = 'Tuio: Current configuration is <%s>' % str(','.join(args))
            Logger.error(err)
            return
        ipport = args[0].split(':')
        if len(ipport) != 2:
            Logger.error('Tuio: Invalid configuration for TUIO provider')
            Logger.error('Tuio: Format must be ip:port (eg. 127.0.0.1:3333)')
            err = 'Tuio: Current configuration is <%s>' % str(','.join(args))
            Logger.error(err)
            return
        self.ip, self.port = args[0].split(':')
        self.port = int(self.port)
        self.handlers = {}
        self.oscid = None
        self.tuio_event_q = deque()
        self.touches = {}

    @staticmethod
    def register(oscpath, classname):
        """Register a new path to handle in TUIO provider"""
        TuioMotionEventProvider.__handlers__[oscpath] = classname

    @staticmethod
    def unregister(oscpath, classname):
        """Unregister a path to stop handling it in the TUIO provider"""
        if oscpath in TuioMotionEventProvider.__handlers__:
            del TuioMotionEventProvider.__handlers__[oscpath]

    @staticmethod
    def create(oscpath, **kwargs):
        """Create a touch event from a TUIO path"""
        if oscpath not in TuioMotionEventProvider.__handlers__:
            raise Exception('Unknown %s touch path' % oscpath)
        return TuioMotionEventProvider.__handlers__[oscpath](**kwargs)

    def start(self):
        """Start the TUIO provider"""
        try:
            from oscpy.server import OSCThreadServer
        except ImportError:
            Logger.info('Please install the oscpy python module to use the TUIO provider.')
            raise
        self.oscid = osc = OSCThreadServer()
        osc.listen(self.ip, self.port, default=True)
        for oscpath in TuioMotionEventProvider.__handlers__:
            self.touches[oscpath] = {}
            osc.bind(oscpath, partial(self._osc_tuio_cb, oscpath))

    def stop(self):
        """Stop the TUIO provider"""
        self.oscid.stop_all()

    def update(self, dispatch_fn):
        """Update the TUIO provider (pop events from the queue)"""
        while True:
            try:
                value = self.tuio_event_q.pop()
            except IndexError:
                return
            self._update(dispatch_fn, value)

    def _osc_tuio_cb(self, oscpath, address, *args):
        self.tuio_event_q.appendleft([oscpath, address, args])

    def _update(self, dispatch_fn, value):
        oscpath, command, args = value
        if command not in [b'alive', b'set']:
            return
        if command == b'set':
            id = args[0]
            if id not in self.touches[oscpath]:
                touch = TuioMotionEventProvider.__handlers__[oscpath](self.device, id, args[1:])
                self.touches[oscpath][id] = touch
                dispatch_fn('begin', touch)
            else:
                touch = self.touches[oscpath][id]
                touch.move(args[1:])
                dispatch_fn('update', touch)
        if command == b'alive':
            alives = args
            to_delete = []
            for id in self.touches[oscpath]:
                if id not in alives:
                    touch = self.touches[oscpath][id]
                    if touch not in to_delete:
                        to_delete.append(touch)
            for touch in to_delete:
                dispatch_fn('end', touch)
                del self.touches[oscpath][touch.id]