from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfStringsOrNone, OneOf,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Wedge
from reportlab.graphics.widgetbase import TypedPropertyCollection
from reportlab.graphics.charts.piecharts import AbstractPieChart, WedgeProperties, _addWedgeLabel, fixLabelOverlaps
from functools import reduce
Make a more complex demo with Label Overlap fixing