import ctypes
class Z3Exception(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)