import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _render_road(self, zoom, translation, angle):
    bounds = PLAYFIELD
    field = [(bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)]
    self._draw_colored_polygon(self.surf, field, self.bg_color, zoom, translation, angle, clip=False)
    grass = []
    for x in range(-20, 20, 2):
        for y in range(-20, 20, 2):
            grass.append([(GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0), (GRASS_DIM * x + 0, GRASS_DIM * y + 0), (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM), (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM)])
    for poly in grass:
        self._draw_colored_polygon(self.surf, poly, self.grass_color, zoom, translation, angle)
    for poly, color in self.road_poly:
        poly = [(p[0], p[1]) for p in poly]
        color = [int(c) for c in color]
        self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)