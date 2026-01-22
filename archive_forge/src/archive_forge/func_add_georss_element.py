from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
def add_georss_element(self, handler, item, w3c_geo=False):
    """Add a GeoRSS XML element using the given item and handler."""
    geom = item.get('geometry')
    if geom is not None:
        if isinstance(geom, (list, tuple)):
            box_coords = None
            if isinstance(geom[0], (list, tuple)):
                if len(geom) == 2:
                    box_coords = geom
                else:
                    raise ValueError('Only should be two sets of coordinates.')
            elif len(geom) == 2:
                self.add_georss_point(handler, geom, w3c_geo=w3c_geo)
            elif len(geom) == 4:
                box_coords = (geom[:2], geom[2:])
            else:
                raise ValueError('Only should be 2 or 4 numeric elements.')
            if box_coords is not None:
                if w3c_geo:
                    raise ValueError('Cannot use simple GeoRSS box in W3C Geo feeds.')
                handler.addQuickElement('georss:box', self.georss_coords(box_coords))
        else:
            gtype = str(geom.geom_type).lower()
            if gtype == 'point':
                self.add_georss_point(handler, geom.coords, w3c_geo=w3c_geo)
            else:
                if w3c_geo:
                    raise ValueError('W3C Geo only supports Point geometries.')
                if gtype in ('linestring', 'linearring'):
                    handler.addQuickElement('georss:line', self.georss_coords(geom.coords))
                elif gtype in ('polygon',):
                    handler.addQuickElement('georss:polygon', self.georss_coords(geom[0].coords))
                else:
                    raise ValueError('Geometry type "%s" not supported.' % geom.geom_type)