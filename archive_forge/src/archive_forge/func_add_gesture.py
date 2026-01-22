import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def add_gesture(self, gesture):
    """Add a new gesture to the database."""
    self.db.append(gesture)