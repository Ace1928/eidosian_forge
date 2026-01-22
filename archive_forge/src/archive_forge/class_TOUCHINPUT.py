import os
class TOUCHINPUT(Structure):
    _fields_ = [('x', LONG), ('y', LONG), ('pSource', HANDLE), ('id', DWORD), ('flags', DWORD), ('mask', DWORD), ('time', DWORD), ('extraInfo', POINTER(ULONG)), ('size_x', DWORD), ('size_y', DWORD)]

    def size(self):
        return (self.size_x, self.size_y)

    def screen_x(self):
        return self.x / 100.0

    def screen_y(self):
        return self.y / 100.0

    def _event_type(self):
        if self.flags & TOUCHEVENTF_MOVE:
            return 'update'
        if self.flags & TOUCHEVENTF_DOWN:
            return 'begin'
        if self.flags & TOUCHEVENTF_UP:
            return 'end'
    event_type = property(_event_type)