import sys
import os
from typing import List
import pygame
import pygame as pg
import pygame.freetype as freetype
class Game:
    """
    A class that handles the game's events, mainloop etc.
    """
    FPS = 50
    SCREEN_WIDTH, SCREEN_HEIGHT = (640, 480)
    BG_COLOR = 'black'

    def __init__(self, caption: str) -> None:
        pg.init()
        self.screen = pg.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pg.display.set_caption(caption)
        self.clock = pg.time.Clock()
        self.print_event = 'showevent' in sys.argv
        self.text_input = TextInput(prompt='> ', pos=(0, 20), screen_dimensions=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT), print_event=self.print_event, text_color='green')

    def main_loop(self) -> None:
        pg.key.start_text_input()
        input_rect = pg.Rect(80, 80, 320, 40)
        pg.key.set_text_input_rect(input_rect)
        while True:
            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    pg.quit()
                    return
            self.text_input.update(events)
            self.screen.fill(self.BG_COLOR)
            self.text_input.draw(self.screen)
            pg.display.update()
            self.clock.tick(self.FPS)