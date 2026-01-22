from kivy.uix.image import Image
from kivy.core.camera import Camera as CoreCamera
from kivy.properties import NumericProperty, ListProperty, \
def _on_index(self, *largs):
    self._camera = None
    if self.index < 0:
        return
    if self.resolution[0] < 0 or self.resolution[1] < 0:
        self._camera = CoreCamera(index=self.index, stopped=True)
    else:
        self._camera = CoreCamera(index=self.index, resolution=self.resolution, stopped=True)
    if self.play:
        self._camera.start()
    self._camera.bind(on_texture=self.on_tex)