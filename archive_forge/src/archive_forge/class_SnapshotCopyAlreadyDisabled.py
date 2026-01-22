from boto.exception import JSONResponseError
class SnapshotCopyAlreadyDisabled(JSONResponseError):
    pass