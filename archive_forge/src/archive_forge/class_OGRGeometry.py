import sys
from binascii import b2a_hex
from ctypes import byref, c_char_p, c_double, c_ubyte, c_void_p, string_at
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.envelope import Envelope, OGREnvelope
from django.contrib.gis.gdal.error import GDALException, SRSException
from django.contrib.gis.gdal.geomtype import OGRGeomType
from django.contrib.gis.gdal.prototypes import geom as capi
from django.contrib.gis.gdal.prototypes import srs as srs_api
from django.contrib.gis.gdal.srs import CoordTransform, SpatialReference
from django.contrib.gis.geometry import hex_regex, json_regex, wkt_regex
from django.utils.encoding import force_bytes
class OGRGeometry(GDALBase):
    """Encapsulate an OGR geometry."""
    destructor = capi.destroy_geom

    def __init__(self, geom_input, srs=None):
        """Initialize Geometry on either WKT or an OGR pointer as input."""
        str_instance = isinstance(geom_input, str)
        if str_instance and hex_regex.match(geom_input):
            geom_input = memoryview(bytes.fromhex(geom_input))
            str_instance = False
        if str_instance:
            wkt_m = wkt_regex.match(geom_input)
            json_m = json_regex.match(geom_input)
            if wkt_m:
                if wkt_m['srid']:
                    srs = int(wkt_m['srid'])
                if wkt_m['type'].upper() == 'LINEARRING':
                    g = capi.create_geom(OGRGeomType(wkt_m['type']).num)
                    capi.import_wkt(g, byref(c_char_p(wkt_m['wkt'].encode())))
                else:
                    g = capi.from_wkt(byref(c_char_p(wkt_m['wkt'].encode())), None, byref(c_void_p()))
            elif json_m:
                g = self._from_json(geom_input.encode())
            else:
                OGRGeomType(geom_input)
                g = capi.create_geom(OGRGeomType(geom_input).num)
        elif isinstance(geom_input, memoryview):
            g = self._from_wkb(geom_input)
        elif isinstance(geom_input, OGRGeomType):
            g = capi.create_geom(geom_input.num)
        elif isinstance(geom_input, self.ptr_type):
            g = geom_input
        else:
            raise GDALException('Invalid input type for OGR Geometry construction: %s' % type(geom_input))
        if not g:
            raise GDALException('Cannot create OGR Geometry from input: %s' % geom_input)
        self.ptr = g
        if srs:
            self.srs = srs
        self.__class__ = GEO_CLASSES[self.geom_type.num]

    def __getstate__(self):
        srs = self.srs
        if srs:
            srs = srs.wkt
        else:
            srs = None
        return (bytes(self.wkb), srs)

    def __setstate__(self, state):
        wkb, srs = state
        ptr = capi.from_wkb(wkb, None, byref(c_void_p()), len(wkb))
        if not ptr:
            raise GDALException('Invalid OGRGeometry loaded from pickled state.')
        self.ptr = ptr
        self.srs = srs

    @classmethod
    def _from_wkb(cls, geom_input):
        return capi.from_wkb(bytes(geom_input), None, byref(c_void_p()), len(geom_input))

    @staticmethod
    def _from_json(geom_input):
        return capi.from_json(geom_input)

    @classmethod
    def from_bbox(cls, bbox):
        """Construct a Polygon from a bounding box (4-tuple)."""
        x0, y0, x1, y1 = bbox
        return OGRGeometry('POLYGON((%s %s, %s %s, %s %s, %s %s, %s %s))' % (x0, y0, x0, y1, x1, y1, x1, y0, x0, y0))

    @staticmethod
    def from_json(geom_input):
        return OGRGeometry(OGRGeometry._from_json(force_bytes(geom_input)))

    @classmethod
    def from_gml(cls, gml_string):
        return cls(capi.from_gml(force_bytes(gml_string)))

    def __or__(self, other):
        """Return the union of the two geometries."""
        return self.union(other)

    def __and__(self, other):
        """Return the intersection of this Geometry and the other."""
        return self.intersection(other)

    def __sub__(self, other):
        """Return the difference this Geometry and the other."""
        return self.difference(other)

    def __xor__(self, other):
        """Return the symmetric difference of this Geometry and the other."""
        return self.sym_difference(other)

    def __eq__(self, other):
        """Is this Geometry equal to the other?"""
        return isinstance(other, OGRGeometry) and self.equals(other)

    def __str__(self):
        """WKT is used for the string representation."""
        return self.wkt

    @property
    def dimension(self):
        """Return 0 for points, 1 for lines, and 2 for surfaces."""
        return capi.get_dims(self.ptr)

    def _get_coord_dim(self):
        """Return the coordinate dimension of the Geometry."""
        return capi.get_coord_dim(self.ptr)

    def _set_coord_dim(self, dim):
        """Set the coordinate dimension of this Geometry."""
        if dim not in (2, 3):
            raise ValueError('Geometry dimension must be either 2 or 3')
        capi.set_coord_dim(self.ptr, dim)
    coord_dim = property(_get_coord_dim, _set_coord_dim)

    @property
    def geom_count(self):
        """Return the number of elements in this Geometry."""
        return capi.get_geom_count(self.ptr)

    @property
    def point_count(self):
        """Return the number of Points in this Geometry."""
        return capi.get_point_count(self.ptr)

    @property
    def num_points(self):
        """Alias for `point_count` (same name method in GEOS API.)"""
        return self.point_count

    @property
    def num_coords(self):
        """Alias for `point_count`."""
        return self.point_count

    @property
    def geom_type(self):
        """Return the Type for this Geometry."""
        return OGRGeomType(capi.get_geom_type(self.ptr))

    @property
    def geom_name(self):
        """Return the Name of this Geometry."""
        return capi.get_geom_name(self.ptr)

    @property
    def area(self):
        """Return the area for a LinearRing, Polygon, or MultiPolygon; 0 otherwise."""
        return capi.get_area(self.ptr)

    @property
    def envelope(self):
        """Return the envelope for this Geometry."""
        return Envelope(capi.get_envelope(self.ptr, byref(OGREnvelope())))

    @property
    def empty(self):
        return capi.is_empty(self.ptr)

    @property
    def extent(self):
        """Return the envelope as a 4-tuple, instead of as an Envelope object."""
        return self.envelope.tuple

    def _get_srs(self):
        """Return the Spatial Reference for this Geometry."""
        try:
            srs_ptr = capi.get_geom_srs(self.ptr)
            return SpatialReference(srs_api.clone_srs(srs_ptr))
        except SRSException:
            return None

    def _set_srs(self, srs):
        """Set the SpatialReference for this geometry."""
        if isinstance(srs, SpatialReference):
            srs_ptr = srs.ptr
        elif isinstance(srs, (int, str)):
            sr = SpatialReference(srs)
            srs_ptr = sr.ptr
        elif srs is None:
            srs_ptr = None
        else:
            raise TypeError('Cannot assign spatial reference with object of type: %s' % type(srs))
        capi.assign_srs(self.ptr, srs_ptr)
    srs = property(_get_srs, _set_srs)

    def _get_srid(self):
        srs = self.srs
        if srs:
            return srs.srid
        return None

    def _set_srid(self, srid):
        if isinstance(srid, int) or srid is None:
            self.srs = srid
        else:
            raise TypeError('SRID must be set with an integer.')
    srid = property(_get_srid, _set_srid)

    def _geos_ptr(self):
        from django.contrib.gis.geos import GEOSGeometry
        return GEOSGeometry._from_wkb(self.wkb)

    @property
    def geos(self):
        """Return a GEOSGeometry object from this OGRGeometry."""
        from django.contrib.gis.geos import GEOSGeometry
        return GEOSGeometry(self._geos_ptr(), self.srid)

    @property
    def gml(self):
        """Return the GML representation of the Geometry."""
        return capi.to_gml(self.ptr)

    @property
    def hex(self):
        """Return the hexadecimal representation of the WKB (a string)."""
        return b2a_hex(self.wkb).upper()

    @property
    def json(self):
        """
        Return the GeoJSON representation of this Geometry.
        """
        return capi.to_json(self.ptr)
    geojson = json

    @property
    def kml(self):
        """Return the KML representation of the Geometry."""
        return capi.to_kml(self.ptr, None)

    @property
    def wkb_size(self):
        """Return the size of the WKB buffer."""
        return capi.get_wkbsize(self.ptr)

    @property
    def wkb(self):
        """Return the WKB representation of the Geometry."""
        if sys.byteorder == 'little':
            byteorder = 1
        else:
            byteorder = 0
        sz = self.wkb_size
        buf = (c_ubyte * sz)()
        capi.to_wkb(self.ptr, byteorder, byref(buf))
        return memoryview(string_at(buf, sz))

    @property
    def wkt(self):
        """Return the WKT representation of the Geometry."""
        return capi.to_wkt(self.ptr, byref(c_char_p()))

    @property
    def ewkt(self):
        """Return the EWKT representation of the Geometry."""
        srs = self.srs
        if srs and srs.srid:
            return 'SRID=%s;%s' % (srs.srid, self.wkt)
        else:
            return self.wkt

    def clone(self):
        """Clone this OGR Geometry."""
        return OGRGeometry(capi.clone_geom(self.ptr), self.srs)

    def close_rings(self):
        """
        If there are any rings within this geometry that have not been
        closed, this routine will do so by adding the starting point at the
        end.
        """
        capi.geom_close_rings(self.ptr)

    def transform(self, coord_trans, clone=False):
        """
        Transform this geometry to a different spatial reference system.
        May take a CoordTransform object, a SpatialReference object, string
        WKT or PROJ, and/or an integer SRID.  By default, return nothing
        and transform the geometry in-place. However, if the `clone` keyword is
        set, return a transformed clone of this geometry.
        """
        if clone:
            klone = self.clone()
            klone.transform(coord_trans)
            return klone
        if isinstance(coord_trans, CoordTransform):
            capi.geom_transform(self.ptr, coord_trans.ptr)
        elif isinstance(coord_trans, SpatialReference):
            capi.geom_transform_to(self.ptr, coord_trans.ptr)
        elif isinstance(coord_trans, (int, str)):
            sr = SpatialReference(coord_trans)
            capi.geom_transform_to(self.ptr, sr.ptr)
        else:
            raise TypeError('Transform only accepts CoordTransform, SpatialReference, string, and integer objects.')

    def _topology(self, func, other):
        """A generalized function for topology operations, takes a GDAL function and
        the other geometry to perform the operation on."""
        if not isinstance(other, OGRGeometry):
            raise TypeError('Must use another OGRGeometry object for topology operations!')
        return func(self.ptr, other.ptr)

    def intersects(self, other):
        """Return True if this geometry intersects with the other."""
        return self._topology(capi.ogr_intersects, other)

    def equals(self, other):
        """Return True if this geometry is equivalent to the other."""
        return self._topology(capi.ogr_equals, other)

    def disjoint(self, other):
        """Return True if this geometry and the other are spatially disjoint."""
        return self._topology(capi.ogr_disjoint, other)

    def touches(self, other):
        """Return True if this geometry touches the other."""
        return self._topology(capi.ogr_touches, other)

    def crosses(self, other):
        """Return True if this geometry crosses the other."""
        return self._topology(capi.ogr_crosses, other)

    def within(self, other):
        """Return True if this geometry is within the other."""
        return self._topology(capi.ogr_within, other)

    def contains(self, other):
        """Return True if this geometry contains the other."""
        return self._topology(capi.ogr_contains, other)

    def overlaps(self, other):
        """Return True if this geometry overlaps the other."""
        return self._topology(capi.ogr_overlaps, other)

    def _geomgen(self, gen_func, other=None):
        """A helper routine for the OGR routines that generate geometries."""
        if isinstance(other, OGRGeometry):
            return OGRGeometry(gen_func(self.ptr, other.ptr), self.srs)
        else:
            return OGRGeometry(gen_func(self.ptr), self.srs)

    @property
    def boundary(self):
        """Return the boundary of this geometry."""
        return self._geomgen(capi.get_boundary)

    @property
    def convex_hull(self):
        """
        Return the smallest convex Polygon that contains all the points in
        this Geometry.
        """
        return self._geomgen(capi.geom_convex_hull)

    def difference(self, other):
        """
        Return a new geometry consisting of the region which is the difference
        of this geometry and the other.
        """
        return self._geomgen(capi.geom_diff, other)

    def intersection(self, other):
        """
        Return a new geometry consisting of the region of intersection of this
        geometry and the other.
        """
        return self._geomgen(capi.geom_intersection, other)

    def sym_difference(self, other):
        """
        Return a new geometry which is the symmetric difference of this
        geometry and the other.
        """
        return self._geomgen(capi.geom_sym_diff, other)

    def union(self, other):
        """
        Return a new geometry consisting of the region which is the union of
        this geometry and the other.
        """
        return self._geomgen(capi.geom_union, other)