import sys
import os
import pygame as pg

        This method handles user input. It returns False when it receives
        a pygame.QUIT event or the user presses escape. The y_offset is
        changed based on mouse and keyboard input. display_fonts() and
        display_surface() use the y_offset to scroll display.
        