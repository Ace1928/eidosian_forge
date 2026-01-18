from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
def test_triplot_bbox_tight():
    """Test triplot with a tight bbox (#1060)."""
    x = np.degrees([-0.101, -0.09, -0.069])
    y = np.degrees([0.872, 0.883, 0.888])
    triangles = np.asarray([[0, 1, 2]])
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.OSGB(approx=False))
    ax.triplot(x, y, triangles, transform=ccrs.Geodetic())
    fig.savefig(BytesIO(), bbox_inches='tight')