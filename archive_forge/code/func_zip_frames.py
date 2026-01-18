from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
from kivy.tests.common import GraphicUnitTest, ensure_web_server
def zip_frames(self, path):
    with ZipFile(path) as zipf:
        return len(zipf.namelist())