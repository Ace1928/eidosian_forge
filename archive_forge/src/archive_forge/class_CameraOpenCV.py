from __future__ import division
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
class CameraOpenCV(CameraBase):
    """
    Implementation of CameraBase using OpenCV
    """
    _update_ev = None

    def __init__(self, **kwargs):
        try:
            self.opencvMajorVersion = int(cv.__version__[0])
        except NameError:
            self.opencvMajorVersion = int(cv2.__version__[0])
        self._device = None
        super(CameraOpenCV, self).__init__(**kwargs)

    def init_camera(self):
        if self.opencvMajorVersion in (3, 4):
            PROPERTY_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            PROPERTY_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            PROPERTY_FPS = cv2.CAP_PROP_FPS
        elif self.opencvMajorVersion == 2:
            PROPERTY_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            PROPERTY_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            PROPERTY_FPS = cv2.cv.CV_CAP_PROP_FPS
        elif self.opencvMajorVersion == 1:
            PROPERTY_WIDTH = cv.CV_CAP_PROP_FRAME_WIDTH
            PROPERTY_HEIGHT = cv.CV_CAP_PROP_FRAME_HEIGHT
            PROPERTY_FPS = cv.CV_CAP_PROP_FPS
        Logger.debug('Using opencv ver.' + str(self.opencvMajorVersion))
        if self.opencvMajorVersion == 1:
            self._device = hg.cvCreateCameraCapture(self._index)
            cv.SetCaptureProperty(self._device, cv.CV_CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cv.SetCaptureProperty(self._device, cv.CV_CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            frame = hg.cvQueryFrame(self._device)
            self._resolution = (int(frame.width), int(frame.height))
            self.fps = cv.GetCaptureProperty(self._device, cv.CV_CAP_PROP_FPS)
        elif self.opencvMajorVersion in (2, 3, 4):
            self._device = cv2.VideoCapture(self._index)
            self._device.set(PROPERTY_WIDTH, self.resolution[0])
            self._device.set(PROPERTY_HEIGHT, self.resolution[1])
            ret, frame = self._device.read()
            self._resolution = (int(frame.shape[1]), int(frame.shape[0]))
            self.fps = self._device.get(PROPERTY_FPS)
        if self.fps == 0 or self.fps == 1:
            self.fps = 1.0 / 30
        elif self.fps > 1:
            self.fps = 1.0 / self.fps
        if not self.stopped:
            self.start()

    def _update(self, dt):
        if self.stopped:
            return
        if self._texture is None:
            self._texture = Texture.create(self._resolution)
            self._texture.flip_vertical()
            self.dispatch('on_load')
        try:
            ret, frame = self._device.read()
            self._format = 'bgr'
            try:
                self._buffer = frame.imageData
            except AttributeError:
                self._buffer = frame.reshape(-1)
            self._copy_to_gpu()
        except:
            Logger.exception("OpenCV: Couldn't get image from Camera")

    def start(self):
        super(CameraOpenCV, self).start()
        if self._update_ev is not None:
            self._update_ev.cancel()
        self._update_ev = Clock.schedule_interval(self._update, self.fps)

    def stop(self):
        super(CameraOpenCV, self).stop()
        if self._update_ev is not None:
            self._update_ev.cancel()
            self._update_ev = None