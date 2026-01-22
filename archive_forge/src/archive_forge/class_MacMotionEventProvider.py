import ctypes
import threading
import collections
import os
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class MacMotionEventProvider(MotionEventProvider):

    def __init__(self, *largs, **kwargs):
        global _instance
        if _instance is not None:
            raise Exception('Only one MacMotionEvent provider is allowed.')
        _instance = self
        super(MacMotionEventProvider, self).__init__(*largs, **kwargs)

    def start(self):
        self.uid = 0
        self.touches = {}
        self.lock = threading.Lock()
        self.queue = collections.deque()
        devices = MultitouchSupport.MTDeviceCreateList()
        num_devices = CFArrayGetCount(devices)
        for i in range(num_devices):
            device = CFArrayGetValueAtIndex(devices, i)
            data_id = str(device)
            self.touches[data_id] = {}
            MTRegisterContactFrameCallback(device, self._mts_callback)
            MTDeviceStart(device, 0)

    def update(self, dispatch_fn):
        try:
            while True:
                event_type, touch = self.queue.popleft()
                dispatch_fn(event_type, touch)
        except:
            pass

    def stop(self):
        pass

    @MTContactCallbackFunction
    def _mts_callback(device, data_ptr, n_fingers, timestamp, frame):
        global _instance
        devid = str(device)
        if devid not in _instance.touches:
            _instance.touches[devid] = {}
        touches = _instance.touches[devid]
        actives = []
        for i in range(n_fingers):
            data = data_ptr[i]
            actives.append(data.identifier)
            data_id = data.identifier
            norm_pos = data.normalized.position
            args = (norm_pos.x, norm_pos.y, data.size)
            if data_id not in touches:
                _instance.lock.acquire()
                _instance.uid += 1
                touch = MacMotionEvent(_instance.device, _instance.uid, args)
                _instance.lock.release()
                _instance.queue.append(('begin', touch))
                touches[data_id] = touch
            else:
                touch = touches[data_id]
                if data.normalized.position.x == touch.sx and data.normalized.position.y == touch.sy:
                    continue
                touch.move(args)
                _instance.queue.append(('update', touch))
        for tid in list(touches.keys())[:]:
            if tid not in actives:
                touch = touches[tid]
                touch.update_time_end()
                _instance.queue.append(('end', touch))
                del touches[tid]
        return 0