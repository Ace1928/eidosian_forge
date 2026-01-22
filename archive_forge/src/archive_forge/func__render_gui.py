from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def _render_gui(self, mode):
    try:
        import pygame
    except ImportError:
        raise DependencyNotInstalled('pygame is not installed, run `pip install gym[toy_text]`')
    if self.window_surface is None:
        pygame.init()
        if mode == 'human':
            pygame.display.init()
            pygame.display.set_caption('Frozen Lake')
            self.window_surface = pygame.display.set_mode(self.window_size)
        elif mode == 'rgb_array':
            self.window_surface = pygame.Surface(self.window_size)
    assert self.window_surface is not None, 'Something went wrong with pygame. This should never happen.'
    if self.clock is None:
        self.clock = pygame.time.Clock()
    if self.hole_img is None:
        file_name = path.join(path.dirname(__file__), 'img/hole.png')
        self.hole_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
    if self.cracked_hole_img is None:
        file_name = path.join(path.dirname(__file__), 'img/cracked_hole.png')
        self.cracked_hole_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
    if self.ice_img is None:
        file_name = path.join(path.dirname(__file__), 'img/ice.png')
        self.ice_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
    if self.goal_img is None:
        file_name = path.join(path.dirname(__file__), 'img/goal.png')
        self.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
    if self.start_img is None:
        file_name = path.join(path.dirname(__file__), 'img/stool.png')
        self.start_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
    if self.elf_images is None:
        elfs = [path.join(path.dirname(__file__), 'img/elf_left.png'), path.join(path.dirname(__file__), 'img/elf_down.png'), path.join(path.dirname(__file__), 'img/elf_right.png'), path.join(path.dirname(__file__), 'img/elf_up.png')]
        self.elf_images = [pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in elfs]
    desc = self.desc.tolist()
    assert isinstance(desc, list), f'desc should be a list or an array, got {desc}'
    for y in range(self.nrow):
        for x in range(self.ncol):
            pos = (x * self.cell_size[0], y * self.cell_size[1])
            rect = (*pos, *self.cell_size)
            self.window_surface.blit(self.ice_img, pos)
            if desc[y][x] == b'H':
                self.window_surface.blit(self.hole_img, pos)
            elif desc[y][x] == b'G':
                self.window_surface.blit(self.goal_img, pos)
            elif desc[y][x] == b'S':
                self.window_surface.blit(self.start_img, pos)
            pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)
    bot_row, bot_col = (self.s // self.ncol, self.s % self.ncol)
    cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
    last_action = self.lastaction if self.lastaction is not None else 1
    elf_img = self.elf_images[last_action]
    if desc[bot_row][bot_col] == b'H':
        self.window_surface.blit(self.cracked_hole_img, cell_rect)
    else:
        self.window_surface.blit(elf_img, cell_rect)
    if mode == 'human':
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])
    elif mode == 'rgb_array':
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))