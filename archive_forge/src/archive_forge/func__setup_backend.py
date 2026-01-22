import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pygame import error
def _setup_backend(backend):
    global list_cameras, Camera
    if backend == 'opencv-mac':
        from pygame import _camera_opencv
        list_cameras = _camera_opencv.list_cameras_darwin
        Camera = _camera_opencv.CameraMac
    elif backend == 'opencv':
        from pygame import _camera_opencv
        list_cameras = _camera_opencv.list_cameras
        Camera = _camera_opencv.Camera
    elif backend in ('_camera (msmf)', '_camera (v4l2)'):
        from pygame import _camera
        list_cameras = _camera.list_cameras
        Camera = _camera.Camera
    elif backend == 'videocapture':
        from pygame import _camera_vidcapture
        warnings.warn('The VideoCapture backend is not recommended and may be removed.For Python3 and Windows 8+, there is now a native Windows backend built into pygame.', DeprecationWarning, stacklevel=2)
        _camera_vidcapture.init()
        list_cameras = _camera_vidcapture.list_cameras
        Camera = _camera_vidcapture.Camera
    else:
        raise ValueError('unrecognized backend name')