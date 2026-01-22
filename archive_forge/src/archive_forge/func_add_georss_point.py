from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
def add_georss_point(self, handler, coords, w3c_geo=False):
    """
        Adds a GeoRSS point with the given coords using the given handler.
        Handles the differences between simple GeoRSS and the more popular
        W3C Geo specification.
        """
    if w3c_geo:
        lon, lat = coords[:2]
        handler.addQuickElement('geo:lat', '%f' % lat)
        handler.addQuickElement('geo:lon', '%f' % lon)
    else:
        handler.addQuickElement('georss:point', self.georss_coords((coords,)))