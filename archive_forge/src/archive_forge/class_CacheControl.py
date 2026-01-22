import re
class CacheControl(object):
    """
    Represents the Cache-Control header.

    By giving a type of ``'request'`` or ``'response'`` you can
    control what attributes are allowed (some Cache-Control values
    only apply to requests or responses).
    """
    update_dict = UpdateDict

    def __init__(self, properties, type):
        self.properties = properties
        self.type = type

    @classmethod
    def parse(cls, header, updates_to=None, type=None):
        """
        Parse the header, returning a CacheControl object.

        The object is bound to the request or response object
        ``updates_to``, if that is given.
        """
        if updates_to:
            props = cls.update_dict()
            props.updated = updates_to
        else:
            props = {}
        for match in token_re.finditer(header):
            name = match.group(1)
            value = match.group(2) or match.group(3) or None
            if value:
                try:
                    value = int(value)
                except ValueError:
                    pass
            props[name] = value
        obj = cls(props, type=type)
        if updates_to:
            props.updated_args = (obj,)
        return obj

    def __repr__(self):
        return '<CacheControl %r>' % str(self)
    max_stale = value_property('max-stale', none='*', type='request')
    min_fresh = value_property('min-fresh', type='request')
    only_if_cached = exists_property('only-if-cached', type='request')
    public = exists_property('public', type='response')
    private = value_property('private', none='*', type='response')
    no_cache = value_property('no-cache', none='*')
    no_store = exists_property('no-store')
    no_transform = exists_property('no-transform')
    must_revalidate = exists_property('must-revalidate', type='response')
    proxy_revalidate = exists_property('proxy-revalidate', type='response')
    max_age = value_property('max-age', none=-1)
    s_maxage = value_property('s-maxage', type='response')
    s_max_age = s_maxage
    stale_while_revalidate = value_property('stale-while-revalidate', type='response')
    stale_if_error = value_property('stale-if-error', type='response')

    def __str__(self):
        return serialize_cache_control(self.properties)

    def copy(self):
        """
        Returns a copy of this object.
        """
        return self.__class__(self.properties.copy(), type=self.type)