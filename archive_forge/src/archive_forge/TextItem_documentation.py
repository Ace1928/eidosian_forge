from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject

        Returns only the anchor point for when calulating view ranges.
        
        Sacrifices some visual polish for fixing issue #2642.
        