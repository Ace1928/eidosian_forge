from pyglet.window import Window
from pyglet.clock import Clock
from threading import Thread, Lock
def __event_loop__(self, **win_args):
    """
        The event loop thread function. Do not override or call
        directly (it is called by __init__).
        """
    gl_lock.acquire()
    try:
        try:
            super().__init__(**self.win_args)
            self.switch_to()
            self.setup()
        except Exception as e:
            print('Window initialization failed: %s' % str(e))
            self.has_exit = True
    finally:
        gl_lock.release()
    clock = Clock()
    clock.fps_limit = self.fps_limit
    while not self.has_exit:
        dt = clock.tick()
        gl_lock.acquire()
        try:
            try:
                self.switch_to()
                self.dispatch_events()
                self.clear()
                self.update(dt)
                self.draw()
                self.flip()
            except Exception as e:
                print('Uncaught exception in event loop: %s' % str(e))
                self.has_exit = True
        finally:
            gl_lock.release()
    super().close()