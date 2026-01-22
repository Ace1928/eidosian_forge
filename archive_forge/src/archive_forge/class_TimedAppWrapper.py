import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
class TimedAppWrapper:

    def __init__(self, app, func):
        self.app = app
        self.app.withdraw()
        self.func = func

    def on_timer(self):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.func())
        except Exception:
            kernel.log.exception('Error in message handler')
        self.app.after(poll_interval, self.on_timer)

    def start(self):
        self.on_timer()
        self.app.mainloop()