import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _create_track(self):
    CHECKPOINTS = 12
    checkpoints = []
    for c in range(CHECKPOINTS):
        noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        alpha = 2 * math.pi * c / CHECKPOINTS + noise
        rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            self.start_alpha = 2 * math.pi * -0.5 / CHECKPOINTS
            rad = 1.5 * TRACK_RAD
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
    self.road = []
    x, y, beta = (1.5 * TRACK_RAD, 0, 0)
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2 * math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break
    i1, i2 = (-1, -1)
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return False
        pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    if self.verbose:
        print('Track generation: %i..%i -> %i-tiles track' % (i1, i2, i2 - i1))
    assert i1 != -1
    assert i2 != -1
    track = track[i1:i2 - 1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    well_glued_together = np.sqrt(np.square(first_perp_x * (track[0][2] - track[-1][2])) + np.square(first_perp_y * (track[0][3] - track[-1][3])))
    if well_glued_together > TRACK_DETAIL_STEP:
        return False
    border = [False] * len(track)
    for i in range(len(track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = track[i - neg - 0][1]
            beta2 = track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]
    for i in range(len(track)):
        alpha1, beta1, x1, y1 = track[i]
        alpha2, beta2, x2, y2 = track[i - 1]
        road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
        road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
        road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
        road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
        vertices = [road1_l, road1_r, road2_r, road2_l]
        self.fd_tile.shape.vertices = vertices
        t = self.world.CreateStaticBody(fixtures=self.fd_tile)
        t.userData = t
        c = 0.01 * (i % 3) * 255
        t.color = self.road_color + c
        t.road_visited = False
        t.road_friction = 1.0
        t.idx = i
        t.fixtures[0].sensor = True
        self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
        self.road.append(t)
        if border[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
            b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1), y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
            b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
            b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2), y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
            self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (255, 255, 255) if i % 2 == 0 else (255, 0, 0)))
    self.track = track
    return True