import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _get_text_specs(self, angle, loc, xylabel):
    """Get rotation and alignments specs for a single label"""
    if angle > 180:
        angle -= 360
    if loc == 'geo':
        loc = self._get_loc_from_angle(angle)
    if not self.rotate_labels:
        kw = {'rotation': 0, 'ha': 'center', 'va': 'center'}
        if loc == 'right':
            kw.update(ha='left')
        elif loc == 'left':
            kw.update(ha='right')
        elif loc == 'top':
            kw.update(va='bottom')
        elif loc == 'bottom':
            kw.update(va='top')
    else:
        if isinstance(self.rotate_labels, (float, int)) and (not isinstance(self.rotate_labels, bool)):
            angle = self.rotate_labels
        kw = {'rotation': angle, 'rotation_mode': 'anchor', 'va': 'center'}
        if angle < 90 + self.offset_angle and angle > -90 + self.offset_angle:
            kw.update(ha='left', rotation=angle)
        else:
            kw.update(ha='right', rotation=angle + 180)
    if getattr(self, xylabel + 'padding') < 0:
        if 'ha' in kw:
            if kw['ha'] == 'left':
                kw['ha'] = 'right'
            elif kw['ha'] == 'right':
                kw['ha'] = 'left'
        if 'va' in kw:
            if kw['va'] == 'top':
                kw['va'] = 'bottom'
            elif kw['va'] == 'bottom':
                kw['va'] = 'top'
    return kw