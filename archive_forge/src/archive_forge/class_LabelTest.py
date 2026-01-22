from kivy.app import runTouchApp
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_random_color
import timeit
import re
import random
from functools import partial
from the brougham.
class LabelTest(GridLayout):
    text = StringProperty(text)
    sized_text = StringProperty(annotated_text)