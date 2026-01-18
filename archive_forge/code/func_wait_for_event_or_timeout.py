from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
from kivy.tests.common import GraphicUnitTest, ensure_web_server
def wait_for_event_or_timeout(self, event):
    timeout = 30 * self.maxfps
    while timeout and (not event.is_set()):
        self.advance_frames(1)
        timeout -= 1