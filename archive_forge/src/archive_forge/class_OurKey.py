from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
class OurKey(Key):
    key_data = KeyData(is_white_key, c_width, c_height, c_down_state_initial, c_down_state_rect_initial, c_notify_down_method, c_notify_up_method, c_updates, c_event_down, c_event_up, c_image_strip, c_event_right_white_down, c_event_right_white_up, c_event_right_black_down, c_event_right_black_up)