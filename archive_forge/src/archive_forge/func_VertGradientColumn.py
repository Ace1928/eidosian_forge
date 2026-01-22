import os
import pygame as pg
def VertGradientColumn(surf, topcolor, bottomcolor):
    """creates a new 3d vertical gradient array"""
    topcolor = np.array(topcolor, copy=False)
    bottomcolor = np.array(bottomcolor, copy=False)
    diff = bottomcolor - topcolor
    width, height = surf.get_size()
    column = np.arange(height, dtype='float') / height
    column = np.repeat(column[:, np.newaxis], [3], 1)
    column = topcolor + (diff * column).astype('int')
    column = column.astype('uint8')[np.newaxis, :, :]
    return pg.surfarray.map_array(surf, column)