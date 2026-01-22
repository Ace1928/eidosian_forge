import os
import signal
import subprocess
import time
from wandb_watchdog.utils import echo, has_attribute
from wandb_watchdog.events import PatternMatchingEventHandler
class AutoRestartTrick(Trick):
    """Starts a long-running subprocess and restarts it on matched events.

    The command parameter is a list of command arguments, such as
    ['bin/myserver', '-c', 'etc/myconfig.ini'].

    Call start() after creating the Trick. Call stop() when stopping
    the process.
    """

    def __init__(self, command, patterns=None, ignore_patterns=None, ignore_directories=False, stop_signal=signal.SIGINT, kill_after=10):
        super(AutoRestartTrick, self).__init__(patterns, ignore_patterns, ignore_directories)
        self.command = command
        self.stop_signal = stop_signal
        self.kill_after = kill_after
        self.process = None

    def start(self):
        self.process = subprocess.Popen(self.command, preexec_fn=os.setsid)

    def stop(self):
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), self.stop_signal)
        except OSError:
            pass
        else:
            kill_time = time.time() + self.kill_after
            while time.time() < kill_time:
                if self.process.poll() is not None:
                    break
                time.sleep(0.25)
            else:
                try:
                    os.killpg(os.getpgid(self.process.pid), 9)
                except OSError:
                    pass
        self.process = None

    @echo.echo
    def on_any_event(self, event):
        self.stop()
        self.start()