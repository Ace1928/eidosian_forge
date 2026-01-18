from functools import reduce
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy import config
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.img_transform as im_trans
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='regrid_image.png', tolerance=5.55)
def test_regrid_image():
    fname = config['repo_data_dir'] / 'raster' / 'natural_earth' / '50-natural-earth-1-downsampled.png'
    nx = 720
    ny = 360
    source_proj = ccrs.PlateCarree()
    source_x, source_y, _ = im_trans.mesh_projection(source_proj, nx, ny)
    data = plt.imread(fname)
    data = data[::-1]
    target_nx = 300
    target_ny = 300
    target_proj = ccrs.InterruptedGoodeHomolosine(emphasis='land')
    target_x, target_y, target_extent = im_trans.mesh_projection(target_proj, target_nx, target_ny)
    new_array = im_trans.regrid(data, source_x, source_y, source_proj, target_proj, target_x, target_y)
    fig = plt.figure(figsize=(10, 10))
    gs = mpl.gridspec.GridSpec(nrows=4, ncols=1, hspace=1.5, wspace=0.5)
    ax = fig.add_subplot(gs[0], projection=target_proj)
    ax.imshow(new_array, origin='lower', extent=target_extent)
    ax.coastlines()
    cmaps = {'red': 'Reds', 'green': 'Greens', 'blue': 'Blues'}
    for i, color in enumerate(['red', 'green', 'blue']):
        ax = fig.add_subplot(gs[i + 1], projection=target_proj)
        ax.imshow(new_array[:, :, i], extent=target_extent, origin='lower', cmap=cmaps[color])
        ax.coastlines()
    gs.tight_layout(fig)
    return fig