from boto.regioninfo import RegionInfo
class SDBRegionInfo(RegionInfo):

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        from boto.sdb.connection import SDBConnection
        super(SDBRegionInfo, self).__init__(connection, name, endpoint, SDBConnection)