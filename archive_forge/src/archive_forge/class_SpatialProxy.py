from django.db.models.query_utils import DeferredAttribute
class SpatialProxy(DeferredAttribute):

    def __init__(self, klass, field, load_func=None):
        """
        Initialize on the given Geometry or Raster class (not an instance)
        and the corresponding field.
        """
        self._klass = klass
        self._load_func = load_func or klass
        super().__init__(field)

    def __get__(self, instance, cls=None):
        """
        Retrieve the geometry or raster, initializing it using the
        corresponding class specified during initialization and the value of
        the field. Currently, GEOS or OGR geometries as well as GDALRasters are
        supported.
        """
        if instance is None:
            return self
        try:
            geo_value = instance.__dict__[self.field.attname]
        except KeyError:
            geo_value = super().__get__(instance, cls)
        if isinstance(geo_value, self._klass):
            geo_obj = geo_value
        elif geo_value is None or geo_value == '':
            geo_obj = None
        else:
            geo_obj = self._load_func(geo_value)
            setattr(instance, self.field.attname, geo_obj)
        return geo_obj

    def __set__(self, instance, value):
        """
        Retrieve the proxied geometry or raster with the corresponding class
        specified during initialization.

        To set geometries, use values of None, HEXEWKB, or WKT.
        To set rasters, use JSON or dict values.
        """
        gtype = self.field.geom_type
        if gtype == 'RASTER' and (value is None or isinstance(value, (str, dict, self._klass))):
            pass
        elif isinstance(value, self._klass):
            if value.srid is None:
                value.srid = self.field.srid
        elif value is None or isinstance(value, (str, memoryview)):
            pass
        else:
            raise TypeError('Cannot set %s SpatialProxy (%s) with value of type: %s' % (instance.__class__.__name__, gtype, type(value)))
        instance.__dict__[self.field.attname] = value
        return value