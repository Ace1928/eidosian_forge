from boto.exception import JSONResponseError
class DBSnapshotNotFound(JSONResponseError):
    pass