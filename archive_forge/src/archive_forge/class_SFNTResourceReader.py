from io import BytesIO
from fontTools.misc.macRes import ResourceReader, ResourceError
class SFNTResourceReader(BytesIO):
    """Simple read-only file wrapper for 'sfnt' resources."""

    def __init__(self, path, res_name_or_index):
        from fontTools import ttLib
        reader = ResourceReader(path)
        if isinstance(res_name_or_index, str):
            rsrc = reader.getNamedResource('sfnt', res_name_or_index)
        else:
            rsrc = reader.getIndResource('sfnt', res_name_or_index)
        if rsrc is None:
            raise ttLib.TTLibError('sfnt resource not found: %s' % res_name_or_index)
        reader.close()
        self.rsrc = rsrc
        super(SFNTResourceReader, self).__init__(rsrc.data)
        self.name = path