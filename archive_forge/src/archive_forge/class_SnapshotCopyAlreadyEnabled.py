from boto.exception import JSONResponseError
class SnapshotCopyAlreadyEnabled(JSONResponseError):
    pass