from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
class LeapMotionListener(Leap.Listener):

    def on_init(self, controller):
        Logger.info('leapmotion: Initialized')

    def on_connect(self, controller):
        Logger.info('leapmotion: Connected')

    def on_disconnect(self, controller):
        Logger.info('leapmotion: Disconnected')

    def on_frame(self, controller):
        frame = controller.frame()
        _LEAP_QUEUE.append(frame)

    def on_exit(self, controller):
        pass