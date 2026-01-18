import pygame as pg
 pygame.examples.setmodescale

On high resolution displays(4k, 1080p) and tiny graphics games (640x480)
show up very small so that they are unplayable. SCALED scales up the window
for you. The game thinks it's a 640x480 window, but really it can be bigger.
Mouse events are scaled for you, so your game doesn't need to do it.

Passing SCALED to pygame.display.set_mode means the resolution depends
on desktop size and the graphics are scaled.
