import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_set_eventstate_on_interactive(self):
    c = self._get_first_controller()
    if not c:
        self.skipTest('No controller connected')
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((400, 400))
    font = pygame.font.Font(None, 20)
    running = True
    screen.fill((255, 255, 255))
    screen.blit(font.render("Press button 'x' (on ps4) or 'a' (on xbox).", True, (0, 0, 0)), (0, 0))
    pygame.display.update()
    controller.set_eventstate(True)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.CONTROLLERBUTTONDOWN:
                running = False
    pygame.display.quit()
    pygame.font.quit()