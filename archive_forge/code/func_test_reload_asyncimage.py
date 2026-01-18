from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
from kivy.tests.common import GraphicUnitTest, ensure_web_server
def test_reload_asyncimage(self):
    from kivy.resources import resource_find
    from kivy.uix.image import AsyncImage
    temp_dir = mkdtemp()
    event = Event()
    image = AsyncImage()
    image.bind(on_load=lambda *args, **kwargs: event.set())
    fn = resource_find('data/logo/kivy-icon-16.png')
    source = join(temp_dir, 'source.png')
    copyfile(fn, source)
    event.clear()
    image.source = source
    self.wait_for_event_or_timeout(event)
    self.render(image, framecount=2)
    self.assertEqual(image.texture_size, [16, 16])
    remove(source)
    fn = resource_find('data/logo/kivy-icon-32.png')
    copyfile(fn, source)
    event.clear()
    image.reload()
    self.wait_for_event_or_timeout(event)
    self.render(image, framecount=2)
    self.assertEqual(image.texture_size, [32, 32])
    remove(source)
    rmtree(temp_dir)